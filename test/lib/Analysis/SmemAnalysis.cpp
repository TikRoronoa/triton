#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

namespace mlir {
namespace test {

static triton::gpu::SwizzledSharedEncodingAttr
getSwizzledEncoding(MLIRContext *ctx, ArrayRef<int64_t> shape,
                    unsigned elemBitWidth,
                    unsigned tRow = 1, unsigned tCol = 32,
                    unsigned sptRow = 1, unsigned sptCol = 1) {
  const unsigned numBanks = 32;
  const unsigned bankBitWidth = 32;
  SmallVector<unsigned> order;
  for (int i = shape.size() - 1; i >= 0; --i)
    order.push_back(i);
  int64_t innerDim = shape.back();
  int elemsPerBank = bankBitWidth / elemBitWidth;
  if (elemsPerBank == 0) elemsPerBank = 1;

  // vec: 每次向量化加载的元素数，最大 128bit
  unsigned vec = std::min((int64_t)(128 / elemBitWidth), innerDim);

  // 尝试不同的 perPhase，找到 conflict 最小的
  unsigned bestPerPhase = 1;
  unsigned bestMaxPhase = numBanks;
  int bestConflict = INT_MAX;

  for (unsigned pp = 1; pp <= 8; pp *= 2) {
    for (unsigned mp = 1; mp <= 16; mp *= 2) {
      // 模拟 warp 访问
      std::vector<int> bankCount(numBanks, 0);
      for (unsigned r = 0; r < tRow; r++) {
        for (unsigned c = 0; c < tCol; c++) {
          int rowIdx = r * sptRow;
          int colIdx = c * sptCol;
          int phase = (rowIdx / pp) % mp;
          int blockNo = colIdx / (mp * (int)vec);
          int combinedPhase = phase ^ blockNo;
          int actualCol = colIdx ^ combinedPhase;
          int bank = (actualCol / elemsPerBank) % numBanks;
          bankCount[bank]++;
        }
      }
      int maxConflict = *std::max_element(bankCount.begin(), bankCount.end());
      if (maxConflict < bestConflict) {
        bestConflict = maxConflict;
        bestPerPhase = pp;
        bestMaxPhase = mp;
      }
      if (bestConflict == 1) break;
    }
    if (bestConflict == 1) break;
  }

  return triton::gpu::SwizzledSharedEncodingAttr::get(
      ctx, vec, bestPerPhase, bestMaxPhase, order,
      triton::gpu::CGAEncodingAttr::get1CTALayout(ctx, shape.size()));
}

// 计算 warp 里每个 thread 访问的 bank
// shape: shared memory 的形状 [rows, cols]
// threadsPerWarp: [row方向线程数, col方向线程数]
// order: 访问顺序
// elemBitWidth: 元素位宽
// 返回：是否有 bank conflict，以及 conflict 的程度
// 计算 swizzled 地址
static int getSwizzledCol(int rowId, int colId,
                           int vec, int perPhase, int maxPhase) {
  int phase = (rowId / perPhase) % maxPhase;
  int blockNo = colId / (maxPhase * vec);
  int combinedPhase = phase ^ blockNo;
  return colId ^ combinedPhase;
}

static std::pair<bool, int>
analyzeWarpBankConflict(ArrayRef<int64_t> shape,
                        ArrayRef<unsigned> threadsPerWarp,
                        ArrayRef<unsigned> sizePerThread,
                        ArrayRef<unsigned> order,
                        unsigned elemBitWidth,
                        // swizzle 参数，0 表示不用 swizzle
                        int vec=0, int perPhase=0, int maxPhase=0) {
  if (shape.size() != 2) return {false, 0};

  const int numBanks = 32;
  const int bankWidth = 4; // bytes per bank

  int elemsPerBank = bankWidth / (elemBitWidth / 8);
  if (elemsPerBank == 0) elemsPerBank = 1;

  int tRow = threadsPerWarp[0];
  int tCol = threadsPerWarp[1];

  std::vector<int> bankCount(numBanks, 0);

  for (int r = 0; r < tRow; r++) {
    for (int c = 0; c < tCol; c++) {
      int rowIdx = r * sizePerThread[0];
      int colIdx = c * sizePerThread[1];

      // 如果有 swizzle，计算 swizzled 地址
      int actualCol = colIdx;
      if (vec > 0 && perPhase > 0 && maxPhase > 0)
        actualCol = getSwizzledCol(rowIdx, colIdx, vec, perPhase, maxPhase);

      int bank = (actualCol / elemsPerBank) % numBanks;
      bankCount[bank]++;
    }
  }

  int maxConflict = *std::max_element(bankCount.begin(), bankCount.end());
  return {maxConflict > 1, maxConflict};
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
        auto memDescTy =
            dyn_cast<triton::gpu::MemDescType>(result.getType());
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

        llvm::outs() << "    Buffer [" << bufSize << " bytes, layout: "
                     << layoutType << "]:\n";

        // 简单 bank conflict 检测（基于 shape）
        bool simpleConflict = false;
        if (layoutType == "none" && shape.size() >= 2)
          simpleConflict = (shape.back() % 32 == 0);

        // Warp-level bank conflict 分析
        // 找所有读取这个 buffer 的 local_load op
        for (auto *user : result.getUsers()) {
          if (user->getName().getStringRef() != "ttg.local_load")
            continue;

          auto loadResult = user->getResult(0);
          auto tensorTy = dyn_cast<RankedTensorType>(loadResult.getType());
          if (!tensorTy) continue;

          auto blockedEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(
              tensorTy.getEncoding());
          if (!blockedEnc) continue;

          auto threadsPerWarp = blockedEnc.getThreadsPerWarp();
          auto sizePerThread = blockedEnc.getSizePerThread();
          auto order = blockedEnc.getOrder();
          unsigned elemBitWidth =
              memDescTy.getElementType().getIntOrFloatBitWidth();

          // 如果是 swizzled layout，传入 swizzle 参数
          int vec = 0, perPhase = 0, maxPhase = 0;
          if (auto swizzled = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(encoding)) {
            vec = swizzled.getVec();
            perPhase = swizzled.getPerPhase();
            maxPhase = swizzled.getMaxPhase();
          }
          auto [hasConflict, conflictDegree] = analyzeWarpBankConflict(
              shape, threadsPerWarp, sizePerThread, order, elemBitWidth,
              vec, perPhase, maxPhase);

          llvm::outs() << "      local_load → tensor"
                       << " threadsPerWarp=[" << threadsPerWarp[0]
                       << "," << threadsPerWarp[1] << "]"
                       << " sizePerThread=[" << sizePerThread[0]
                       << "," << sizePerThread[1] << "]\n";
          llvm::outs() << "      warp-level bank conflict: "
                       << (hasConflict ? "YES ⚠️ (" : "NO (")
                       << conflictDegree << "-way)\n";

          // 如果是 swizzled 但还有 conflict，建议更好的参数
          if (hasConflict && layoutType == "swizzled") {
            // 搜索更好的 swizzle 参数
            unsigned elemBW = memDescTy.getElementType().getIntOrFloatBitWidth();
            int elemsPerBank2 = 4 / (elemBW / 8);
            if (elemsPerBank2 == 0) elemsPerBank2 = 1;
            int64_t innerDim2 = shape.back();
            unsigned bestVec = std::min((int64_t)(128 / elemBW), innerDim2);
            unsigned bestPP = 1, bestMP = 1;
            int bestConflict2 = conflictDegree;
            for (unsigned pp = 1; pp <= 8; pp *= 2) {
              for (unsigned mp = 1; mp <= 16; mp *= 2) {
                std::vector<int> bc(32, 0);
                for (unsigned r2 = 0; r2 < threadsPerWarp[0]; r2++) {
                  for (unsigned c2 = 0; c2 < threadsPerWarp[1]; c2++) {
                    int ri = r2 * sizePerThread[0];
                    int ci = c2 * sizePerThread[1];
                    int ph = (ri / pp) % mp;
                    int bn = ci / (mp * (int)bestVec);
                    int cp = ph ^ bn;
                    int ac = ci ^ cp;
                    bc[(ac / elemsPerBank2) % 32]++;
                  }
                }
                int mc = *std::max_element(bc.begin(), bc.end());
                if (mc < bestConflict2) {
                  bestConflict2 = mc;
                  bestPP = pp;
                  bestMP = mp;
                }
                if (bestConflict2 == 1) break;
              }
              if (bestConflict2 == 1) break;
            }
            if (bestConflict2 < conflictDegree) {
              llvm::outs() << "      → current swizzle suboptimal for this access pattern\n";
              llvm::outs() << "      → suggest: vec=" << bestVec
                           << ", perPhase=" << bestPP
                           << ", maxPhase=" << bestMP
                           << " (would reduce to " << bestConflict2 << "-way)\n";
            }
          }
        }

        if (simpleConflict) {
          llvm::outs() << "      shape-level risk: YES ⚠️ (inner dim % 32 == 0)\n";
          llvm::outs() << "      → queued for auto-fix\n";
          opsToFix.push_back(op);
        }
      });
    });

    // 修复阶段
    for (auto *op : opsToFix) {
      auto result = op->getResult(0);
      auto memDescTy =
          dyn_cast<triton::gpu::MemDescType>(result.getType());
      if (!memDescTy) continue;

      auto shape = memDescTy.getShape();
      unsigned elemBitWidth =
          memDescTy.getElementType().getIntOrFloatBitWidth();

      // 找 local_load 的 BlockedEncoding 来优化 swizzle 参数
      unsigned tRow = 1, tCol = 32, sptRow = 1, sptCol = 1;
      for (auto *user : result.getUsers()) {
        if (user->getName().getStringRef() != "ttg.local_load") continue;
        auto tensorTy = dyn_cast<RankedTensorType>(user->getResult(0).getType());
        if (!tensorTy) continue;
        auto enc = dyn_cast<triton::gpu::BlockedEncodingAttr>(tensorTy.getEncoding());
        if (!enc) continue;
        tRow = enc.getThreadsPerWarp()[0];
        tCol = enc.getThreadsPerWarp()[1];
        sptRow = enc.getSizePerThread()[0];
        sptCol = enc.getSizePerThread()[1];
        break;
      }
      auto newEncoding = getSwizzledEncoding(
          op->getContext(), shape, elemBitWidth,
          tRow, tCol, sptRow, sptCol);
      auto newMemDescTy = triton::gpu::MemDescType::get(
          shape, memDescTy.getElementType(), newEncoding,
          memDescTy.getMemorySpace(), memDescTy.getMutableMemory());

      SmallVector<Operation *> deallocsToFix;
      for (auto *user : result.getUsers())
        if (user->getName().getStringRef() == "ttg.local_dealloc")
          deallocsToFix.push_back(user);

      OpBuilder builder(op);
      auto newAlloc = triton::gpu::LocalAllocOp::create(
          builder, op->getLoc(), newMemDescTy, op->getOperands());

      result.replaceAllUsesWith(newAlloc.getResult());

      for (auto *dealloc : deallocsToFix) {
        OpBuilder b(dealloc);
        triton::gpu::LocalDeallocOp::create(
            b, dealloc->getLoc(), newAlloc.getResult());
        dealloc->erase();
      }

      op->erase();

      // 修复后重新分析验证
      unsigned tRow2 = 1, tCol2 = 32, sptRow2 = 1, sptCol2 = 1;
      for (auto *user : newAlloc.getResult().getUsers()) {
        if (user->getName().getStringRef() != "ttg.local_load") continue;
        auto tensorTy = dyn_cast<RankedTensorType>(user->getResult(0).getType());
        if (!tensorTy) continue;
        auto enc = dyn_cast<triton::gpu::BlockedEncodingAttr>(tensorTy.getEncoding());
        if (!enc) continue;
        tRow2 = enc.getThreadsPerWarp()[0];
        tCol2 = enc.getThreadsPerWarp()[1];
        sptRow2 = enc.getSizePerThread()[0];
        sptCol2 = enc.getSizePerThread()[1];
        break;
      }
      SmallVector<unsigned> newOrder;
      for (int i = shape.size() - 1; i >= 0; --i)
        newOrder.push_back(i);
      auto [newHasConflict, newConflictDegree] = analyzeWarpBankConflict(
          shape, ArrayRef<unsigned>{tRow2, tCol2},
          ArrayRef<unsigned>{sptRow2, sptCol2},
          newOrder, elemBitWidth,
          newEncoding.getVec(), newEncoding.getPerPhase(),
          newEncoding.getMaxPhase());

      llvm::outs() << "  → fixed: swizzled<vec="
                   << newEncoding.getVec()
                   << ", perPhase=" << newEncoding.getPerPhase()
                   << ", maxPhase=" << newEncoding.getMaxPhase() << ">\n";
      llvm::outs() << "  → after fix: bank_conflict="
                   << (newHasConflict ? "YES ⚠️ (" : "NO (")
                   << newConflictDegree << "-way)\n";
    }

    llvm::outs() << "\n==============================\n";
  }
};

void registerSmemAnalysisPass() {
  PassRegistration<SmemAnalysisPass>();
}

} // namespace test
} // namespace mlir
