#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace test {

static triton::gpu::SwizzledSharedEncodingAttr
getSwizzledEncoding(MLIRContext *ctx, ArrayRef<int64_t> shape,
                    unsigned elemBitWidth) {
  const unsigned numBanks = 32;
  const unsigned bankBitWidth = 32;
  // order must match shape rank
  SmallVector<unsigned> order;
  for (int i = shape.size() - 1; i >= 0; --i)
    order.push_back(i);
  int64_t innerDim = shape.back();
  int elemsPerBankRow = (numBanks * bankBitWidth) / elemBitWidth;
  unsigned vec = std::min((int64_t)(128 / elemBitWidth), innerDim);
  unsigned perPhase = std::max(1, elemsPerBankRow / (int)innerDim);
  unsigned maxPhase = std::max(1u, (unsigned)(numBanks / perPhase));
  return triton::gpu::SwizzledSharedEncodingAttr::get(
      ctx, vec, perPhase, maxPhase, order,
      triton::gpu::CGAEncodingAttr::get1CTALayout(ctx, shape.size()));
}

struct SmemAnalysisPass
    : public PassWrapper<SmemAnalysisPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SmemAnalysisPass)

  StringRef getArgument() const override { return "smem-analysis"; }
  StringRef getDescription() const override {
    return "Analyze and fix shared memory bank conflict risks";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ModuleAllocation allocation(moduleOp);

    llvm::outs() << "\n=== Shared Memory Analysis ===\n";

    SmallVector<Operation *> opsToFix;

    moduleOp.walk([&](FunctionOpInterface funcOp) {
      auto *alloc = allocation.getFuncData(funcOp);
      size_t totalSize = alloc->getSharedMemorySize();

      llvm::outs() << "\nKernel: " << funcOp.getName() << "\n";
      llvm::outs() << "  Total shared memory: " << totalSize << " bytes\n";
      llvm::outs() << "  Buffers:\n";

      funcOp.walk([&](Operation *op) {
        if (op->getName().getStringRef() != "ttg.local_alloc")
          return;

        auto result = op->getResult(0);
        auto memDescTy = dyn_cast<triton::gpu::MemDescType>(result.getType());
        if (!memDescTy) return;

        auto encoding = memDescTy.getEncoding();
        auto shape = memDescTy.getShape();

        auto bufferIds = alloc->getBufferIds(result);
        size_t bufSize = 0;
        for (auto id : bufferIds)
          bufSize += alloc->getAllocatedSize(id);

        StringRef layoutType = "none";
        if (triton::gpu::isPaddedEncoding(encoding))
          layoutType = "padded";
        else if (isa<triton::gpu::SwizzledSharedEncodingAttr>(encoding))
          layoutType = "swizzled";

        bool conflict = false;
        if (layoutType == "none" && shape.size() >= 2)
          conflict = (shape.back() % 32 == 0);

        llvm::outs() << "    - size: " << bufSize << " bytes"
                     << ", layout: " << layoutType
                     << ", bank_conflict_risk: "
                     << (conflict ? "YES ⚠️" : "NO") << "\n";

        if (conflict) {
          llvm::outs() << "      → queued for auto-fix\n";
          opsToFix.push_back(op);
        }
      });
    });

    // 修复阶段
    for (auto *op : opsToFix) {
      auto result = op->getResult(0);
      auto memDescTy = dyn_cast<triton::gpu::MemDescType>(result.getType());
      if (!memDescTy) continue;

      auto shape = memDescTy.getShape();
      unsigned elemBitWidth =
          memDescTy.getElementType().getIntOrFloatBitWidth();

      auto newEncoding = getSwizzledEncoding(
          op->getContext(), shape, elemBitWidth);
      auto newMemDescTy = triton::gpu::MemDescType::get(
          shape, memDescTy.getElementType(), newEncoding,
          memDescTy.getMemorySpace(), memDescTy.getMutableMemory());

      // 收集所有 dealloc users
      SmallVector<Operation *> deallocsToFix;
      for (auto *user : result.getUsers())
        if (user->getName().getStringRef() == "ttg.local_dealloc")
          deallocsToFix.push_back(user);

      // 创建新 alloc
      OpBuilder builder(op);
      auto newAlloc = triton::gpu::LocalAllocOp::create(
          builder, op->getLoc(), newMemDescTy, op->getOperands());

      // 替换所有 use
      result.replaceAllUsesWith(newAlloc.getResult());

      // 替换 dealloc
      for (auto *dealloc : deallocsToFix) {
        OpBuilder b(dealloc);
        triton::gpu::LocalDeallocOp::create(
            b, dealloc->getLoc(), newAlloc.getResult());
        dealloc->erase();
      }

      op->erase();

      llvm::outs() << "  → fixed: vec=" << newEncoding.getVec()
                   << ", perPhase=" << newEncoding.getPerPhase()
                   << ", maxPhase=" << newEncoding.getMaxPhase() << "\n";
    }

    llvm::outs() << "\n==============================\n";
  }
};

void registerSmemAnalysisPass() {
  PassRegistration<SmemAnalysisPass>();
}

} // namespace test
} // namespace mlir
