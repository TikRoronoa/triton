#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace test {

struct SmemAnalysisPass
    : public PassWrapper<SmemAnalysisPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SmemAnalysisPass)

  StringRef getArgument() const override { return "smem-analysis"; }
  StringRef getDescription() const override {
    return "Analyze shared memory usage and bank conflict risks";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ModuleAllocation allocation(moduleOp);

    llvm::outs() << "\n=== Shared Memory Analysis ===\n";

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

        // 判断 layout 类型
        StringRef layoutType = "none";
        if (triton::gpu::isPaddedEncoding(encoding))
          layoutType = "padded";
        else if (isa<triton::gpu::SwizzledSharedEncodingAttr>(encoding))
          layoutType = "swizzled";

        // 判断 bank conflict 风险
        // 你的经验：内层维度是 32 的倍数时有风险
        bool conflict = false;
        if (layoutType == "none" && shape.size() >= 2)
          conflict = (shape.back() % 32 == 0);

        llvm::outs() << "    - size: " << bufSize << " bytes"
                     << ", layout: " << layoutType
                     << ", bank_conflict_risk: "
                     << (conflict ? "YES ⚠️" : "NO") << "\n";

        if (conflict)
          llvm::outs() << "      → consider swizzled or padded layout\n";
      });
    });

    llvm::outs() << "\n==============================\n";
  }
};

void registerSmemAnalysisPass() {
  PassRegistration<SmemAnalysisPass>();
}

} // namespace test
} // namespace mlir
