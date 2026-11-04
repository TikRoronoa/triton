#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "triton-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

namespace {

/////////////////////////////
// UTILS
/////////////////////////////

int getSelfLatencyFromAttr(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  auto helper = TritonDialect::getLoaded(module)->getSelfLatencyAttrHelper();
  if (!helper.isAttrPresent(op))
    return 0;
  int val = helper.getAttr(op).getInt();
  helper.removeAttr(op);
  return val;
}

// Check if the load can be pipelined entirely in shared memory,
// or if we need to load to registers.
bool mustLoadToRegisters(Operation *op) {
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    // AsyncCopyGlobalToLocalOp does not support the non-zero "other" value.
    // With consumer consuming directly the shared memory, there would be no way
    // to replace masked values with the "other" value.
    if (loadOp.getOther() && !isZeroConst(loadOp.getOther()))
      return true;
  }

  if (!op->hasOneUse())
    return true;
  Operation *user = *op->getUsers().begin();
  auto alloc = dyn_cast<ttg::LocalAllocOp>(user);
  if (!alloc)
    return true;

  Attribute loadEncoding;
  if (auto descLoad = dyn_cast<DescriptorLoadLikeOpInterface>(op)) {
    auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    loadEncoding = nvidia_gpu::getEncodingFromDescriptor(op, tensorType,
                                                         descLoad.getDesc());
  }
  return loadEncoding && (loadEncoding != alloc.getType().getEncoding());
}

int getDefUseStageDiff(Operation *op, scf::ForOp forOp,
                       CoarseSchedule &schedule) {
  assert(schedule.count(op) && "Op not found in the schedule");
  int defStage = schedule[op].first;
  CoarseSchedule::Cluster defCluster = schedule[op].second;
  std::optional<int> useStage;
  DenseSet<Operation *> topLevelUsers =
      triton::getTopLevelUsersInLoop(op, forOp);
  // Special case for loads used by local_alloc:
  // we must consider the uses of the local_alloc, as it may be removed and its
  // uses will become direct uses of the async load.
  // TODO: This is overly conservative, we may need to restrict to cases where
  // local_alloc is used by a dot product and has correct encoding.
  if (isa<tt::LoadOp, tt::DescriptorLoadLikeOpInterface>(op)) {
    DenseSet<Operation *> allocUsers;
    for (Operation *topLevelUser : topLevelUsers) {
      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(topLevelUser)) {
        DenseSet<Operation *> users =
            triton::getTopLevelUsersInLoop(localAlloc, forOp);
        allocUsers.insert(users.begin(), users.end());
      }
    }
    topLevelUsers.insert(allocUsers.begin(), allocUsers.end());
  }
  DenseSet<Operation *> topLevelWaitUsers;
  for (Operation *topLevelUser : topLevelUsers) {
    if (isa<ttng::WaitBarrierOp>(topLevelUser)) {
      topLevelWaitUsers.insert(topLevelUser);
    }
  }
  for (Operation *topLevelUser : topLevelUsers) {
    int _useStage = schedule[topLevelUser].first;
    CoarseSchedule::Cluster _useCluster = schedule[topLevelUser].second;
    if (*_useCluster > *defCluster) {
      // Check if we need extra buffer due to unusual execution order
      // The issue occurs when users of the load are scheduled in a later
      // cluster, which happens when conditional code gets moved to epilogue
      // cluster. This creates a race condition where the local load happens
      // after the global-to-local copy for the next pipeline stage starts.
      _useStage++;
    }
    useStage = std::min(_useStage, useStage.value_or(_useStage));
  }
  // Waits tells us the buffer is still in use until the wait completes, we
  // can't simply load from the buffer and replace the uses of the buffer with
  // the load. The stage diff needs to account for the furthest wait.
  for (Operation *topLevelUser : topLevelWaitUsers) {
    int _useStage = schedule[topLevelUser].first;
    useStage = std::max(_useStage, useStage.value_or(_useStage));
  }
  if (!useStage)
    return 0;
  assert(useStage >= defStage && "Op used before defined");
  return useStage.value() - defStage;
}

void replaceAllUsesDominatedBy(Operation *domOp, Value newValue, Value oldValue,
                               DominanceInfo &domInfo) {
  if (newValue == oldValue)
    return;
  oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
    return domInfo.properlyDominates(domOp, use.getOwner());
  });
}

/////////////////////////////
// LOWER LOADS
/////////////////////////////

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, Operation *loadOp,
                         ttg::SharedEncodingTrait sharedEnc,
                         unsigned distance) {
  return triton::createAlloc(
      forOp, cast<RankedTensorType>(loadOp->getResultTypes().front()),
      loadOp->getLoc(), sharedEnc, distance);
}

void createAsyncCopy(scf::ForOp forOp, tt::LoadOp loadOp, Value alloc,
                     Value insertIdx, Value extractIdx, int contiguity,
                     CoarseSchedule &schedule) {
  OpBuilderForStage builder(loadOp.getLoc(), forOp, schedule);

  //返回第一个使用load结果的op，这里就是convert_layout
  Operation *firstUse = getFirstUseOfPipelinedOp({loadOp}, forOp, schedule);
  assert(firstUse && "LoadOp has no users");
  // Replace the load with async copy, wait and loal_load.
  OpBuilder::InsertionGuard guard(builder);//作用域结束时恢复 builder 插入点
  builder.setInsertionPoint(loadOp); //新 op 插在 loadOp 前面
  builder.setStageCluster(schedule[loadOp]);//后续创建的 op 继承 load 的 stage/cluster（GEMM 里 stage=0, cluster=3
  Value src = loadOp.getPtr();//从原 load 取出指针、mask、other，后面交给 AsyncCopyGlobalToLocal
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();

  // Create async copy
  //从多缓冲 alloc（形状 N × tile）里，用 insertIdx 切出当前要写入的那一槽。（这里N=2，tile=128x32）
  Value view = createSingleBufferView(builder, alloc, insertIdx);
  /*
  这两行创建 ttg.async_copy_global_to_local：
  用原 tt.load 的语义，但目标改成 shared 上的 view（异步，对应 Ampere cp.async）
  返回值是 async token，下一行 AsyncCommitGroupOp 把它收成一组，之后才能 async_wait。
  %view = ttg.memdesc_index %allocA, %insertIdx
  %token = ttg.async_copy_global_to_local %src, %view ...
  %commit = ttg.async_commit_group %token
  */
  Operation *copy = ttg::AsyncCopyGlobalToLocalOp::create(
      builder, 
      src, //原 load 的指针
      view, //当前写槽（memdesc_index(alloc, insertIdx)）
      mask, other,//mask，other与原 load 相同 
      loadOp.getCache(), loadOp.getEvict(),//原 load 属性原样带上
      loadOp.getIsVolatile(), 
      contiguity);//向量化宽度提示（前面 canBeConvertedToAsyncLoad / axis 分析算出的）
  Operation *commit =
      ttg::AsyncCommitGroupOp::create(builder, copy->getResult(0));

  // Create wait and local load
  //后面插的 op 改用 消费者 的拍（GEMM A：convert 的 stage=2），不再跟 load 的 stage=0
  builder.setStageCluster(schedule[firstUse]);
  /*
  这两行创建 ttg.async_wait 和 ttg.local_load：
  等 commit 结束（组内所有 copy 完成），再从 shared 上读取当前槽（异步，对应 Ampere cp.wait）
  %wait = ttg.async_wait %commit
  %viewLoad = ttg.memdesc_index %allocA, %extractIdx
  %localLoad = ttg.local_load %viewLoad ...
  // stage 0: copy + commit（前面已设）
  // stage 2（firstUse）:
  %wait = ttg.async_wait %commit {num = 0}
  %viewLoad = ttg.memdesc_index %allocA, %extractIdx
  // 接着 local_load %viewLoad, %wait → 替换原 load 的 uses
  */
  auto wait = ttg::AsyncWaitOp::create(builder, commit->getResult(0), 0);
  auto viewLoad = createSingleBufferView(builder, alloc, extractIdx);

  if (!loadOp.getOther() || isZeroConst(loadOp.getOther())) {//GEMM走这个分支，因为A的mask恒为0
    // If masking isn't required, load directly from shared
    /*
    函数会新建：
    %ll = ttg.local_load %viewLoad, %waitToken
        : !ttg.memdesc<128x32xf16, ...> -> tensor<128x32xf16, #blocked>
    其他使用load的结果的op，都会被替换成这个新的local_load
    // 之前
    %12 = ttg.convert_layout %11 ...
    // 之后
    %12 = ttg.convert_layout %ll ...

    */
    replaceUsesWithLocalLoad(builder, 
                            loadOp->getResult(0), //load的结果（tensor<128x32xf16>）
                            viewLoad, //当前读槽（memdesc_index(alloc, extractIdx)）（tensor<128x32xf16>），本地load的指针
                            wait.getResult()); //wait token
  } else if (loadOp->use_begin() != loadOp->use_end()) {
    // Otherwise, create a select for non-zero other values as they are not
    // handled by AsyncCopyGlobalToLocalOp for now.
    auto sharedLoad = ttg::LocalLoadOp::create(builder, loadOp.getType(),
                                               viewLoad, wait.getResult());
    auto select = arith::SelectOp::create(
        builder, loadOp.getType(),
        // Use the mask operand from the original load, not the one with a
        // potentially transformed layout.
        loadOp.getMask(), sharedLoad.getResult(), other);
    loadOp->replaceAllUsesWith(select->getResults());
  }
  schedule.erase(loadOp);
  loadOp->erase();
}

void createTMAAsyncCopy(
    scf::ForOp forOp, Operation *loadOp, Value desc, Value alloc,
    Value insertIdx, Value extractIdx, Value barrier, Operation *waitOp,
    CoarseSchedule &schedule,
    function_ref<void(OpBuilderForStage &, Value, Value, Value, Value)>
        createCopy) {
  OpBuilderForStage builder(loadOp->getLoc(), forOp, schedule);

  Operation *firstUse = getFirstUseOfPipelinedOp({loadOp}, forOp, schedule);
  assert(firstUse && "LoadOp has no users");
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());

  builder.setInsertionPoint(loadOp);
  builder.setStageCluster(schedule[loadOp]);

  // Create async copy
  Value view = createSingleBufferView(builder, alloc, insertIdx);

  Value pred = arith::ConstantIntOp::create(builder, 1, 1);
  createCopy(builder, desc, barrier, view, pred);

  // Create local load after the wait
  builder.setInsertionPointAfter(waitOp);
  builder.setStageCluster(schedule[firstUse]);
  auto viewLoad = createSingleBufferView(builder, alloc, extractIdx);
  replaceUsesWithLocalLoad(builder, loadOp->getResult(0), viewLoad);

  schedule.erase(loadOp);
  loadOp->erase();
}

void createTMAAsyncLoad(scf::ForOp forOp, tt::DescriptorLoadOp loadOp,
                        Value alloc, Value insertIdx, Value extractIdx,
                        Value barrier, Operation *waitOp,
                        CoarseSchedule &schedule) {
  return createTMAAsyncCopy(forOp, loadOp, loadOp.getDesc(), alloc, insertIdx,
                            extractIdx, barrier, waitOp, schedule,
                            [&](OpBuilderForStage &builder, Value desc,
                                Value barrier, Value view, Value pred) {
                              ttng::AsyncTMACopyGlobalToLocalOp::create(
                                  builder, loadOp.getLoc(), desc,
                                  loadOp.getIndices(), barrier, view, pred);
                            });
}

void createTMAAsyncGather(scf::ForOp forOp, tt::DescriptorGatherOp gatherOp,
                          Value alloc, Value insertIdx, Value extractIdx,
                          Value barrier, Operation *waitOp,
                          CoarseSchedule &schedule) {
  return createTMAAsyncCopy(
      forOp, gatherOp, gatherOp.getDesc(), alloc, insertIdx, extractIdx,
      barrier, waitOp, schedule,
      [&](OpBuilderForStage &builder, Value desc, Value barrier, Value view,
          Value pred) {
        Value xOffsets = ttng::sextI16ToI32Indices(gatherOp.getXOffsets(),
                                                   builder, gatherOp.getLoc());
        ttng::AsyncTMAGatherOp::create(builder, gatherOp.getLoc(), desc,
                                       xOffsets, gatherOp.getYOffset(), barrier,
                                       view, pred);
      });
}

struct AsyncLoad {
  int stageDiff;
  int contiguity = 1;
  Value alloc;
  Value barrier;
  Operation *waitOp;
  SharedEncodingTrait sharedEncoding;
};
struct LoadGroupInfo {
  Value insertIdx;
  Value extractIdx;
  Value phase;
  bool hasTMALoad = false;
};

// Convert a scalar load to a load of a tensor of shape <1>.
void convertScalarToTensorLoad(Operation *op, CoarseSchedule &schedule,
                               scf::ForOp forOp) {
  auto scalarLoad = cast<tt::LoadOp>(op);
  Type scalarTy = scalarLoad.getType();
  OpBuilderForStage builder(op->getLoc(), op, schedule);
  builder.setInsertionPoint(op);
  MLIRContext *ctx = op->getContext();
  auto nWarps = lookupNumWarps(op);
  ModuleOp mod = forOp->getParentOfType<ModuleOp>();
  auto threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
  auto numCTAs = TritonGPUDialect::getNumCTAs(mod);
  auto blockedEnc =
      getDefaultBlockedEncoding(ctx, {1}, nWarps, threadsPerWarp, numCTAs);
  auto newPtrTy =
      RankedTensorType::get({1}, scalarLoad.getPtr().getType(), blockedEnc);
  auto newPtr =
      tt::SplatOp::create(builder, op->getLoc(), newPtrTy, scalarLoad.getPtr());
  scalarLoad.getPtrMutable().assign(newPtr);
  if (scalarLoad.getMask()) {
    auto newMaskTy =
        RankedTensorType::get({1}, scalarLoad.getMask().getType(), blockedEnc);
    auto newMask = tt::SplatOp::create(builder, op->getLoc(), newMaskTy,
                                       scalarLoad.getMask());
    scalarLoad.getMaskMutable().assign(newMask);
  }
  if (scalarLoad.getOther()) {
    auto newOtherTy =
        RankedTensorType::get({1}, scalarLoad.getOther().getType(), blockedEnc);
    auto newOther = tt::SplatOp::create(builder, op->getLoc(), newOtherTy,
                                        scalarLoad.getOther());
    scalarLoad.getOtherMutable().assign(newOther);
  }
  auto newDstTy = RankedTensorType::get({1}, scalarLoad.getType(), blockedEnc);
  scalarLoad.getResult().setType(newDstTy);
  builder.setInsertionPointAfter(op);
  Operation *firstUse = getFirstUseOfPipelinedOp({op}, forOp, schedule);
  builder.setStageCluster(schedule[firstUse]);
  Operation *unsplat = tt::UnsplatOp::create(builder, op->getLoc(), scalarTy,
                                             scalarLoad.getResult());
  scalarLoad.getResult().replaceAllUsesExcept(unsplat->getResult(0), unsplat);
}

void createTMABarrierAndWait(
    scf::ForOp forOp, llvm::MapVector<Operation *, AsyncLoad> &asyncLoads,
    const llvm::MapVector<int, LoadGroupInfo> &loadGroups,
    CoarseSchedule &schedule) {
  SmallVector<SmallVector<Operation *>> commonWaitGroups;
  llvm::SmallDenseSet<Operation *> visited;
  // Find groups of loads that can share the same barrier. We look consecutive
  // loads and check that there are uses in between.
  for (auto &[loadOp, asyncLoad] : asyncLoads) {
    if (!isTMALoad(loadOp) || visited.count(loadOp))
      continue;
    llvm::SmallDenseSet<Operation *> users;
    SmallVector<Operation *> group;
    Block *loadBlock = loadOp->getBlock();
    auto addToGroup = [&](Operation *loadOp) {
      group.push_back(loadOp);
      visited.insert(loadOp);
      for (Operation *user : loadOp->getUsers()) {
        // Special case for MMAv3 loads, we can ignore the alloc and only
        // consider uses of the alloc op since it will be removed.
        if (!mustLoadToRegisters(loadOp)) {
          assert(loadOp->hasOneUse());
          auto alloc = cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
          if (alloc->getBlock() == loadBlock) {
            users.insert(alloc->getUsers().begin(), alloc->getUsers().end());
            continue;
          }
        }
        Operation *userInBlock = loadBlock->findAncestorOpInBlock(*user);
        if (userInBlock)
          users.insert(userInBlock);
      }
    };
    addToGroup(loadOp);
    Operation *nextOp = loadOp->getNextNode();
    int numBuffers = asyncLoad.stageDiff;
    while (nextOp) {
      if (users.count(nextOp) || visited.count(nextOp))
        break;
      if (isTMALoad(nextOp) && asyncLoads.count(nextOp)) {
        if (asyncLoads[nextOp].stageDiff != numBuffers)
          break;
        if (group.size() > 0 && schedule[group[0]] == schedule[nextOp]) {
          addToGroup(nextOp);
        }
      }
      nextOp = nextOp->getNextNode();
    }
    commonWaitGroups.push_back(group);
  }

  // For each group calculate the size and insert the barrier after the last
  // load.
  for (SmallVector<Operation *> &group : commonWaitGroups) {
    int sizeInBytes = 0;
    int numBuffers = asyncLoads[group[0]].stageDiff;
    const LoadGroupInfo loadGroup = loadGroups.find(numBuffers)->second;
    for (Operation *op : group) {
      auto tensorTy = cast<RankedTensorType>(op->getResultTypes()[0]);
      int loadSize = product(getShapePerCTA(tensorTy));
      sizeInBytes += loadSize * tensorTy.getElementTypeBitWidth() / 8;
    }

    Value barrierAlloc = triton::createBarrierAlloc(forOp, numBuffers);
    OpBuilderForStage builder(forOp.getLoc(), group[0], schedule);
    Value barrier = triton::createSingleBufferView(builder, barrierAlloc,
                                                   loadGroup.insertIdx);
    Value pred = arith::ConstantIntOp::create(builder, 1, 1);
    ttng::BarrierExpectOp::create(builder, barrier, sizeInBytes, pred);

    builder.setInsertionPointAfter(group.back());
    Operation *firstUse = getFirstUseOfPipelinedOp(group, forOp, schedule);
    builder.setStageCluster(schedule[firstUse]);
    Value barrierViewWait = triton::createSingleBufferView(
        builder, barrierAlloc, loadGroup.extractIdx);
    auto wait =
        ttng::WaitBarrierOp::create(builder, barrierViewWait, loadGroup.phase);

    // Update the async loads info.
    for (Operation *op : group) {
      asyncLoads[op].barrier = barrier;
      asyncLoads[op].waitOp = wait;
    }
  }
}

// Check if load requires additional buffer for a mma pipelining
bool loadRequiresAdditionalBuffer(Operation *loadOp) {
  std::function<void(Operation *, SmallVector<Operation *> & out)>
      collectNonViewUsers = [&](Operation *op, SmallVector<Operation *> &out) {
        for (Operation *user : op->getUsers()) {
          if (user->hasTrait<OpTrait::MemDescViewTrait>())
            collectNonViewUsers(user, out);
          else
            out.push_back(user);
        }
      };
  // Pattern match the op sequence used for loading mmav3 operands
  if (!mustLoadToRegisters(loadOp)) {
    assert(loadOp->hasOneUse());
    ttg::LocalAllocOp alloc =
        dyn_cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
    if (alloc) {
      SmallVector<Operation *> nonViewUsers;
      collectNonViewUsers(alloc, nonViewUsers);
      return llvm::any_of(nonViewUsers, [&](Operation *op) {
        return isa<ttng::WarpGroupDotOp>(op);
      });
    }
  }
  return false;
}

scf::ForOp lowerLoads(scf::ForOp forOp, CoarseSchedule &schedule,
                      triton::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  llvm::MapVector<Operation *, AsyncLoad> asyncLoads;
  llvm::MapVector<int, LoadGroupInfo> loadGroups;
  llvm::SmallVector<Operation *> scalarLoads;
  // Only visit the top level ops, we do not support pipelining conditional
  // loads for now
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<tt::LoadOp, tt::DescriptorLoadLikeOpInterface>(op)) {//只看普通 tt.load 和 TMA 类 load，GEMM 是前者
      // GEMM：load 在 stage 0，convert/dot 在 stage 2 → stageDiff = 2。
      int stageDiff = getDefUseStageDiff(&op, forOp, schedule);//消费者的 stage − load 的 stage
      if (stageDiff == 0) {//同一 stage，不考虑
        // Don't care about non-pipelined loads. Scalar loads will be converted
        // to tensor loads if they are pipelined.
        continue;
      }
      // 判断load是否可以走cp.async
      SharedEncodingTrait sharedEncoding;
      bool canUseAsyncCp = false;
      int contiguity = 1;
      if (!isa<RankedTensorType>(op.getResultTypes()[0])) {// 结果不是 ranked tensor → 标量 load
        // cp.async 对太窄的类型不划算/不好用，至少 32 bit（f32/i32/i64 等）才考虑异步。
        canUseAsyncCp = op.getResultTypes()[0].getIntOrFloatBitWidth() >= 32;
        auto numCTAs = lookupNumCTAs(forOp);
        sharedEncoding = ttg::SwizzledSharedEncodingAttr::get(
            forOp.getContext(), 1, 1, 1, {0},
            ttg::CGAEncodingAttr::get1DLayout(forOp.getContext(), numCTAs));
        if (canUseAsyncCp) {
          scalarLoads.push_back(&op);
        }
      } else {// tensor load
        // 这里是 tensor load（GEMM 即 tt.load）
        // 选出 从GMEM load 到 SMEM 的 shared encoding（#ttg.swizzled_shared）
        // 后面 createAlloc 会用它生成!ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>里面的 #shared 的 encoding
        sharedEncoding = getSharedEncoding(&op);
        // Do not create async loads for small loads (cp.async requires at least
        // 4 bytes)
        canUseAsyncCp =
            isa<tt::LoadOp>(op) &&
            canBeConvertedToAsyncLoad(cast<tt::LoadOp>(op), axisInfoAnalysis);// true
        // 寄存器 layout → shared layout 拷贝时，每线程实际能连续写多少 字节
        int copyVecBytes = getCopyVecBytes(
            cast<RankedTensorType>(op.getResultTypes()[0]), sharedEncoding);
        // 看 写入 SMEM 时 的向量宽：cp.async 至少 4 字节
        canUseAsyncCp &= copyVecBytes >= 4;
        if (canUseAsyncCp) {
          auto loadOp = cast<tt::LoadOp>(op);
          auto ptr = loadOp.getPtr();
          unsigned vec = axisInfoAnalysis.getContiguity(ptr);
          if (auto mask = loadOp.getMask())
            vec = std::min<unsigned>(vec,
                                     axisInfoAnalysis.getMaskAlignment(mask));
          contiguity = vec;
        }
      }
      if (canUseAsyncCp || isTMALoad(&op)) {
        if (loadRequiresAdditionalBuffer(&op)) {// for Hopper
          // Allocate additional buffer required by the wgmma pipelining.
          stageDiff += 1;
        }
        // asyncLoads 记录所有 load 的 stageDiff、contiguity、sharedEncoding
        /*
        llvm::MapVector<Operation *, AsyncLoad> asyncLoads;
        struct AsyncLoad {
          int stageDiff;
          int contiguity = 1;
          Value alloc;
          Value barrier;
          Operation *waitOp;
          SharedEncodingTrait sharedEncoding;
        };
        */
        auto &asyncLoad = asyncLoads[&op];
        asyncLoad.stageDiff = stageDiff;
        asyncLoad.contiguity = contiguity;
        asyncLoad.sharedEncoding = sharedEncoding;//#ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
      } else if (stageDiff > 1) {
        // Distance-1 loads can in most cases be pipelined in registers without
        // any performance degradation, as the schedule will usually reorder the
        // user and the producer so there is no liverange overlap, and no copy
        // needed.
        op.emitRemark() << "Pipelining load that cannot use vectorized "
                           "copy. This will likely "
                           "lead to pipelining in registers and severe "
                           "performance degradation.";
      }
    }
  }

  // Convert scalar loads to be able to use async copy.
  for (auto op : scalarLoads) {// GEMM会跳过这里
    convertScalarToTensorLoad(op, schedule, forOp);
  }

  if (asyncLoads.empty())
    return forOp;

  for (auto &[loadOp, asyncLoad] : asyncLoads) {
    /*
    asyncLoad.stageDiff = 2
    createAlloc 把 distance 插到 shape 最前面，以A为例：
    !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
                 ↑
        distance = stageDiff
    运行时用 insertIdx / extractIdx（0 或 1）做 memdesc_index，
    一轮写 slot0、用 slot1，下一轮对调，避免 producer/consumer 踩同一块。
    */
    // %10 = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    // createAlloc会在forOp之前插入local_alloc
    Value alloc = createAlloc(forOp, loadOp, asyncLoad.sharedEncoding,
                              asyncLoad.stageDiff);
    asyncLoad.alloc = alloc;
    /*
    insert({2, {}})：为深度 2 建一个空的 LoadGroupInfo（A、B 的 stageDiff 都是 2 → 进同一组，只建一次）
    */
    loadGroups.insert({asyncLoad.stageDiff, {}});
    if (isTMALoad(loadOp)) {
      loadGroups[asyncLoad.stageDiff].hasTMALoad = true;
    }
  }
  //下面准备rewrite这个forOp，从这里开始改 forOp
  IRRewriter builder(forOp);
  builder.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  // Create a counter to index into the allocations per loop iteration.
  // NOTE: We create two duplicates values, insertIdx and extractIdx so that the
  // pipeliner will re-materialize the value in later stages of the pipeline
  // instead of carrying it as a dependency across multiple iterations.
  Value minusOne = arith::ConstantIntOp::create(builder, loc, -1, 32);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  SmallVector<Value> newOperands;
  /*
  拿到forOp的参数个数：1个循环变量和3个变量
  scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(...)
          ^^^^^
     induction var
  %arg0 / %arg1 / %arg2：下界、上界、步长
  %arg5：每轮 += step 的那个（常叫 iv）
  */
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  for (auto [_, loadGroup] : loadGroups) {
    newOperands.push_back(minusOne); // insertIdx，写SMEM的idx，初始值是-1
    newOperands.push_back(minusOne); // extractIdx，读SMEM的idx，初始值是-1
    if (loadGroup.hasTMALoad) {
      // A single barrier arrival sequence is a "phase" and two phases can
      // overlap, provided the phases are differentiated with an alternating
      // boolean value.
      newOperands.push_back(zero); // phase
    }
  }

  // Patch the loop to add the new loop carried dependencies.
  // 用 newOperands（-1, -1）重建一个带更多 iter_args 的 scf.for，旧 loop 删掉。
  // 主体 ops 没换逻辑，只是换了个带更多 iter_args 的新 scf.for，并把旧 body 搬过去。
  forOp = addIterArgsToLoop(builder, forOp, newOperands);

  // Update yield op with temporary yield values
  /*
  addIterArgsToLoop 刚加了新 iter_args，scf.yield 也必须多 yield 同样个数，否则非法。
  scf.for 的约定是：
  iter_args 个数 = yield 操作数个数 = for 的 result 个数
  每一轮结束，yield 的值变成下一轮的 iter_args；循环结束后又是 for 的返回值。少 yield 一个，类型/SSA 对不上，IR 非法。
  */
  auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (unsigned i = 0; i < newOperands.size(); ++i) {
    forYield.getResultsMutable().append(newOperands[i]);
  }

  builder.setInsertionPoint(forOp);
  loc = forOp.getLoc();
  int argIdx = newOperandIndex;
  /*
  struct LoadGroupInfo {
    Value insertIdx;     //4
    Value extractIdx;    //5
    Value phase;//nullptr        // 仅 TMA 时才有
    bool hasTMALoad;
  };
  */
  for (auto &[numBuffers, loadGroup] : loadGroups) {//loadGroups有且只有一个item：{2, LoadGroupInfo}
    // 从刚挂上的 iter_args 里把 insert/extract 下标取出来。
    Value insertIdx = forOp.getBody()->getArgument(argIdx);//argIdx是4， insertIdx是-1
    argIdx++;
    Value extractIdx = forOp.getBody()->getArgument(argIdx);//argIdx是5， extractIdx是-1
    argIdx++;
    Value phase = nullptr;
    if (loadGroup.hasTMALoad) {
      phase = forOp.getBody()->getArgument(argIdx);
      argIdx++;
    }

    // Create two counters for the insert and extract indices to avoid creating
    // long liverange.
    //在 body 最前面插 IR。这样后面的 async_copy / local_load 才能用到本轮刚算出来的 idx。
    /*
    ^bb:
      // ← 这里插入 idx 更新，就是 insertIdx 和 extractIdx 的更新
      ... async_copy / wait / local_load / dot / addptr ...
      scf.yield ...
    */
    builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());

    /*
    这行把 更新后的 insert 下标 写回 loadGroup：
    loadGroup.insertIdx = (旧 insertIdx + 1) % numBuffers //(-1 + 1) % 2 = 0
    右边的 insertIdx：本轮从 iter_args 取的旧值（第一轮是 -1）
    左边：新 Value（第一轮是 0），后面给 async_copy 的 memdesc_index，并 yield 给下一轮
    */
    // 这两个createIncrementModulo 生成“每轮更新下标”的代码
    Value numBuffersVal =
        arith::ConstantIntOp::create(builder, loc, numBuffers, 32); //numBuffers是2
    loadGroup.insertIdx = createIncrementModulo(builder, loc, insertIdx, //insertIdx是-1
                                                numBuffersVal, zero, one); //第一轮是(-1 + 1) % 2 = 0
    Value cndExt = nullptr;
    loadGroup.extractIdx = createIncrementModulo(
        builder, loc, extractIdx, numBuffersVal, zero, one, &cndExt);//第一轮是(-1 + 1) % 2 = 0
    /*

    scf.for %arg5 = %lb to %ub step %step
        iter_args(%arg6 = %ptrA, %arg7 = %ptrB, %arg8 = %acc,
                  %arg9 = %c-1_i32,   // insertIdx  初值
                  %arg10 = %c-1_i32)  // extractIdx 初值
        -> (...) {

      // ---- createIncrementModulo(insert) ----
      %c2_i32 = arith.constant 2 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32

      %0 = arith.addi %arg9, %c1_i32 : i32          // -1+1 / 0+1 / 1+1
      %1 = arith.cmpi sge, %0, %c2_i32 : i32        // >= 2 ?
      %insert = arith.select %1, %c0_i32, %0 : i32  // 绕回 0，否则用 +1

      // ---- createIncrementModulo(extract) ----
      %2 = arith.addi %arg10, %c1_i32 : i32
      %3 = arith.cmpi sge, %2, %c2_i32 : i32        // 这就是 cndExt
      %extract = arith.select %3, %c0_i32, %2 : i32

      // 后面：memdesc_index %alloc[%insert] / [%extract]
      // ...
      scf.yield %..., %..., %..., %insert, %extract : ...
    }
    没有真正的 % 指令，就是 addi + cmpi sge + select。
    每轮跑一遍，yield 把 %insert/%extract 传回下一轮的 %arg9/%arg10。

    */
    if (phase) {//GEMM不会进这里
      Value nextPhase = arith::XOrIOp::create(builder, loc, phase, one);
      phase = arith::SelectOp::create(builder, loc, cndExt, nextPhase, phase);
      loadGroup.phase = phase;
    }
  }

  createTMABarrierAndWait(forOp, asyncLoads, loadGroups, schedule);

  /*
  loadGroups[asyncLoad.stageDiff]; 是 LoadGroupInfo
  struct LoadGroupInfo {
    Value insertIdx;
    Value extractIdx;
    Value phase;//nullptr        // 仅 TMA 时才有
    bool hasTMALoad;
  };
  此时，insertIdx是0，extractIdx是0，phase是nullptr，hasTMALoad是false
  */
  bool hasAsyncLoads = false;
  for (auto [op, asyncLoad] : asyncLoads) {//两个load
    auto [insertIdx, extractIdx, phase, _] = loadGroups[asyncLoad.stageDiff];
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                      asyncLoad.contiguity, schedule);
      hasAsyncLoads = true;
    } else if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(op)) {
      createTMAAsyncLoad(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                         asyncLoad.barrier, asyncLoad.waitOp, schedule);
    } else if (auto loadOp = dyn_cast<tt::DescriptorGatherOp>(op)) {
      createTMAAsyncGather(forOp, loadOp, asyncLoad.alloc, insertIdx,
                           extractIdx, asyncLoad.barrier, asyncLoad.waitOp,
                           schedule);
    }
  }
  // Patch the yield with the updated counters. Subtract to account for the loop
  // counter.
  argIdx = newOperandIndex - 1;
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    forYield.setOperand(argIdx++, loadGroup.insertIdx);
    forYield.setOperand(argIdx++, loadGroup.extractIdx);
    if (loadGroup.phase)
      forYield.setOperand(argIdx++, loadGroup.phase);
  }

  // Automatically discover dependencies and schedule new insert/extract ops to
  // correct stages.
  scheduleDependencies(forOp, schedule);

  if (hasAsyncLoads) {
    // Insert sync point for any possibly outstanding loads after the loop. This
    // can happen as we speculatively execute loads in the loop.
    builder.setInsertionPointAfter(forOp);
    ttg::AsyncWaitOp::create(builder, loc, ValueRange({}), 0);
  }

  // Make sure all ops have attributes.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!schedule.count(&op)) {
      op.emitError() << "op not found in the schedule";
    }
    assert(schedule.count(&op) && "op not found in the schedule");
  }
  return forOp;
}

/////////////////////////////
// LOWER MMA
/////////////////////////////

std::pair<Operation *, Operation *>
getTmemUseStageBoundOps(Value alloc, scf::ForOp forOp,
                        CoarseSchedule &schedule) {
  std::pair<Operation *, Operation *> bounds = {nullptr, nullptr};
  for (auto user : alloc.getUsers()) {
    if (!forOp->isAncestor(user->getParentOp())) {
      continue;
    }
    auto topLevelUser = forOp.getBody()->findAncestorOpInBlock(*user);
    if (!bounds.first) {
      bounds.first = topLevelUser;
    }
    if (!bounds.second) {
      bounds.second = topLevelUser;
    }
    if (schedule.isOpBefore(topLevelUser, bounds.first)) {
      bounds.first = topLevelUser;
    }
    if (schedule.isOpBefore(bounds.second, topLevelUser)) {
      bounds.second = topLevelUser;
    }
  }
  return bounds;
}

Operation *hoistBufferOutOfLoop(scf::ForOp forOp, Operation *op,
                                CoarseSchedule &schedule) {
  Operation *newStore = nullptr;
  if (!isa<ttng::TMEMAllocOp, ttg::LocalAllocOp>(op))
    return nullptr;
  // If the alloc is already out of the loop, there is nothing to do.
  if (!forOp->isAncestor(op))
    return nullptr;
  OpBuilderForStage builder(op->getLoc(), forOp, schedule);
  auto allocType = dyn_cast<MemDescType>(op->getResult(0).getType());
  auto newType = triton::gpu::MemDescType::get(
      allocType.getShape(), allocType.getElementType(), allocType.getEncoding(),
      allocType.getMemorySpace(),
      /*mutableMemory=*/true);
  auto newAlloc = builder.clone(*op);
  newAlloc->getResult(0).setType(newType);
  builder.setStageCluster(schedule[op]);
  if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(newAlloc)) {
    tmemAlloc.getSrcMutable().clear();
    builder.setInsertionPointAfter(op);
    Value trueVal = arith::ConstantIntOp::create(builder, 1, 1);
    newStore = ttng::TMEMStoreOp::create(builder, tmemAlloc.getResult(),
                                         op->getOperand(0), trueVal);
  } else {
    auto localAlloc = cast<ttg::LocalAllocOp>(newAlloc);
    localAlloc.getSrcMutable().clear();
    builder.setInsertionPointAfter(op);
    newStore = ttg::LocalStoreOp::create(builder, op->getOperand(0),
                                         localAlloc.getResult());
  }
  replaceUsesAndPropagateType(builder, op, newAlloc->getResult(0));
  op->erase();
  return newStore;
}

void createBarrierAndWaitOps(scf::ForOp forOp, CoarseSchedule &schedule,
                             ttng::MMAv5OpInterface mma, int mmaSelfLatency,
                             Value alloc, int phaseArgIdx,
                             int barrierIdxArgIdx) {
  auto isLoadToBePipelined = [&](Operation *op) {
    return schedule[mma].first > schedule[op].first;
  };

  llvm::SmallDenseSet<Operation *> syncCandidates;

  for (auto user : alloc.getUsers()) {
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (load->getBlock() != mma->getBlock()) {
        continue;
      }
      syncCandidates.insert(load);
    }
  }

  ttng::MMAv5PipelineableOperandsHelper mmaPipeHelper(mma, forOp,
                                                      isLoadToBePipelined);

  for (auto def : mmaPipeHelper.unpipelineableOperandDefs) {
    auto newStore = hoistBufferOutOfLoop(forOp, def, schedule);
    // If the operands are not pipelineable, we need to consider the stores as
    // well.
    if (!mmaPipeHelper.isPipelineable &&
        mmaPipeHelper.isOperandsStateDetermined) {
      if (newStore) {
        syncCandidates.insert(newStore);
      } else {
        syncCandidates.insert(def);
      }
    }
  }

  // Find the first sync candidate that appears after the MMA
  // in the linearized schedule. This is either the first op to appear
  // after the MMA or the first op
  auto linearizedSchedule = schedule.linearized(forOp, mma);
  std::optional<Operation *> latestSyncPoint = linearizedSchedule.findNext(
      [&](Operation *op) { return syncCandidates.contains(op); });

  int mainWaitStage = schedule[mma].first + mmaSelfLatency;
  CoarseSchedule::Cluster mainWaitCluster = schedule[mma].second;
  if (latestSyncPoint && mmaPipeHelper.isOperandsStateDetermined) {
    if (schedule.isOpBefore(*latestSyncPoint, mma)) {
      mainWaitStage = schedule[mma].first + 1;
      mainWaitCluster = schedule.clusters.newBefore(
          schedule.splitClusterBefore(*latestSyncPoint, forOp));
    } else {
      mainWaitStage = schedule[*latestSyncPoint].first;
      mainWaitCluster = schedule.clusters.newBefore(
          schedule.splitClusterBefore(*latestSyncPoint, forOp));
    }
  }

  int numStages = mainWaitStage - schedule[mma].first + 1;

  OpBuilderForStage builder(mma.getLoc(), mma, schedule);
  Value barrierAlloc = createBarrierAlloc(forOp, numStages);
  Value vTrue = arith::ConstantIntOp::create(builder, 1, 1);
  Value phase = forOp.getRegionIterArg(phaseArgIdx);
  Value zero = arith::ConstantIntOp::create(builder, forOp.getLoc(), 0, 32);
  Value barrierIdx;
  if (numStages > 1) {
    barrierIdx = forOp.getRegionIterArg(barrierIdxArgIdx);
  } else {
    barrierIdx = zero;
  }
  Value one = arith::ConstantIntOp::create(builder, forOp.getLoc(), 1, 32);
  Value numStagesVal =
      arith::ConstantIntOp::create(builder, forOp.getLoc(), numStages, 32);

  Value barrierSlice =
      triton::createSingleBufferView(builder, barrierAlloc, barrierIdx);
  mma.addCompletionBarrier(barrierSlice, vTrue);
  mma.setIsAsync(true);

  // List of buffers that may be used until wait completes
  SmallVector<Value> waitBuffers;
  auto mmaAsDotOp = cast<DotOpInterface>(mma.getOperation());
  waitBuffers.push_back(mmaAsDotOp.getA());
  waitBuffers.push_back(mmaAsDotOp.getB());
  if (auto mmaAsScaledDotOp =
          dyn_cast<ttng::TCGen5MMAScaledOp>(mma.getOperation())) {
    waitBuffers.push_back(mmaAsScaledDotOp.getAScale());
    waitBuffers.push_back(mmaAsScaledDotOp.getBScale());
  }

  builder.setInsertionPointAfter(mma);
  builder.setStageCluster({mainWaitStage, mainWaitCluster});
  ttng::WaitBarrierOp::create(builder, barrierSlice, phase, waitBuffers);

  // Add waits before loads in conditional blocks
  for (auto user : alloc.getUsers()) {
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (load->getBlock() == mma->getBlock()) {
        continue;
      }
      auto topLevelUser = forOp.getBody()->findAncestorOpInBlock(*load);
      if (!topLevelUser) {
        continue;
      }
      auto [loadStage, loadCluster] = schedule[topLevelUser];
      if (loadStage < mainWaitStage) {
        builder.setStageCluster({loadStage, loadCluster});
        builder.setInsertionPoint(load);
        ttng::WaitBarrierOp::create(builder, barrierSlice, phase, waitBuffers);
      }
    }
  }

  builder.setStageCluster(schedule[mma]);
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  builder.setInsertionPoint(yieldOp);
  Value newPhase = arith::XOrIOp::create(builder, phase, one);
  Value newBarrierIdx = barrierIdx;
  if (numStages > 1) {
    Value barWrap;
    newBarrierIdx = createIncrementModulo(builder, builder.getLoc(), barrierIdx,
                                          numStagesVal, zero, one, &barWrap);
    newPhase = arith::SelectOp::create(builder, phase.getType(), barWrap,
                                       newPhase, phase);
  }
  yieldOp->replaceUsesOfWith(phase, newPhase);
  yieldOp->replaceUsesOfWith(barrierIdx, newBarrierIdx);
}

void multibufferTensorMemory(scf::ForOp forOp, CoarseSchedule &schedule,
                             ttng::TMEMAllocOp alloc, int bufIdxArgIdx,
                             int tmemUseNumStages) {
  DominanceInfo domInfo(forOp);
  Value bufIdx = forOp.getRegionIterArg(bufIdxArgIdx);
  SmallVector<std::pair<Operation *, Value>> bufIdxDefs;
  auto getCurrBufIdx = [&](Operation *op) {
    for (auto [_op, _val] : llvm::reverse(bufIdxDefs)) {
      if (domInfo.properlyDominates(_op, op)) {
        return _val;
      }
    }
    return Value();
  };
  bufIdxDefs.push_back({&forOp.getBody()->front(), bufIdx});

  OpBuilderForStage builder(alloc.getLoc(), alloc, schedule);
  auto newAlloc = createTMemAlloc(builder, alloc, true, tmemUseNumStages);
  Value numStagesVal =
      arith::ConstantIntOp::create(builder, tmemUseNumStages, 32);
  Value zero = arith::ConstantIntOp::create(builder, 0, 32);
  Value one = arith::ConstantIntOp::create(builder, 1, 32);

  bool multibufferingIsValid = false;

  SmallVector<Operation *> allocUsers =
      llvm::to_vector(alloc.getResult().getUsers());
  auto auxBuilder = OpBuilder(forOp);
  Value replTok = ub::PoisonOp::create(auxBuilder, forOp.getLoc(),
                                       builder.getType<AsyncTokenType>());
  if (newAlloc.getToken()) {
    newAlloc.getToken().replaceAllUsesWith(replTok);
  }
  for (auto user : allocUsers) {
    if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
      store.getDepMutable().clear();
      store.getToken().replaceAllUsesWith(replTok);
      if (forOp->isAncestor(store)) {
        // We can multibuffer, since the store is a point where we can
        // change the buffer index
        multibufferingIsValid = true;
        builder.setStageCluster(schedule[store]);
        builder.setInsertionPoint(store);
        // Change the buffer index to the new buffer index on store.
        Value curBufIdx = getCurrBufIdx(store);
        Value newBufIdx = createIncrementModulo(
            builder, forOp.getLoc(), curBufIdx, numStagesVal, zero, one);
        if (Value pred = store.getPred()) {
          newBufIdx = arith::SelectOp::create(builder, newBufIdx.getType(),
                                              pred, newBufIdx, curBufIdx);
        }
        replaceAllUsesDominatedBy(store, newBufIdx, curBufIdx, domInfo);
        bufIdxDefs.push_back({store, newBufIdx});
        auto tmemSlice =
            triton::createSingleBufferView(builder, newAlloc, newBufIdx);
        store.getDstMutable().assign(tmemSlice);
      } else {
        // Store before the loop
        assert(store->isBeforeInBlock(forOp) && "Store is not before the loop");
        builder.setInsertionPoint(store);
        auto tmemSlice =
            triton::createSingleBufferView(builder, newAlloc, zero);
        store.getDstMutable().assign(tmemSlice);
      }
    } else if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      load.getDepMutable().clear();
      load.getToken().replaceAllUsesWith(replTok);
      if (forOp->isAncestor(load)) {
        builder.setStageCluster(schedule[load]);
        builder.setInsertionPoint(load);
        Value curBufIdx = getCurrBufIdx(load);
        auto tmemSlice =
            triton::createSingleBufferView(builder, newAlloc, curBufIdx);
        load.getSrcMutable().assign(tmemSlice);
      } else {
        // Load after the loop
        assert(forOp->isBeforeInBlock(load) && "Load is not after the loop");
        builder.setInsertionPoint(load);
        auto tmemSlice = triton::createSingleBufferView(
            builder, newAlloc, forOp->getResult(bufIdxArgIdx));
        load.getSrcMutable().assign(tmemSlice);
      }
    } else if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      mma.getAccDepMutable().clear();
      mma.getToken().replaceAllUsesWith(replTok);
      builder.setStageCluster(schedule[mma]);
      builder.setInsertionPoint(mma);
      // We can legally switch to next buffer index if the mma does not use the
      // accumulator
      auto isConstTrue = [](Value v) {
        if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
          if (auto attr = dyn_cast<BoolAttr>(constOp.getValueAttr())) {
            return attr.getValue();
          }
        }
        return false;
      };
      multibufferingIsValid = !isConstTrue(mma.useAccumulator());
      Value curBufIdx = getCurrBufIdx(mma.getOperation());
      Value newBufIdx = createIncrementModulo(
          builder, forOp.getLoc(), curBufIdx, numStagesVal, zero, one);
      newBufIdx =
          arith::SelectOp::create(builder, newBufIdx.getType(),
                                  mma.useAccumulator(), curBufIdx, newBufIdx);
      replaceAllUsesDominatedBy(mma.getOperation(), newBufIdx, curBufIdx,
                                domInfo);
      bufIdxDefs.push_back({mma.getOperation(), newBufIdx});
      auto tmemSlice =
          triton::createSingleBufferView(builder, newAlloc, newBufIdx);
      mma.setAccumulator(tmemSlice);
    } else {
      llvm::errs() << "Unsupported user of the accumulator: " << *user << "\n";
      llvm::report_fatal_error("Unsupported user of the accumulator");
    }
  }
  if (!multibufferingIsValid) {
    llvm::report_fatal_error(
        "Trying to multibuffer TMEM while there is no store to the "
        "accumulator, and the mma uses the accumulator all the time.");
  }
  alloc.getToken().replaceAllUsesWith(newAlloc.getToken());
  alloc->erase();

  Value newBufIdx = bufIdxDefs.back().second;
  replaceAllUsesDominatedBy(newBufIdx.getDefiningOp(), newBufIdx, bufIdx,
                            domInfo);
}

scf::ForOp lowerMMA(ttng::MMAv5OpInterface mma, scf::ForOp forOp,
                    CoarseSchedule &schedule) {
  auto isLoadToBePipelined = [&](Operation *op) {
    return schedule[mma].first > schedule[op].first;
  };
  Value alloc = mma.getAccumulator();

  int mmaSelfLatency = getSelfLatencyFromAttr(mma.getOperation());
  if (mmaSelfLatency == 0) {
    return forOp;
  }

  // Create barrier and wait ops
  std::pair<Operation *, Operation *> tmemUseStageBoundOps =
      getTmemUseStageBoundOps(alloc, forOp, schedule);
  int tmemUseNumStages = schedule[tmemUseStageBoundOps.second].first -
                         schedule[tmemUseStageBoundOps.first].first;
  // If def is in the earlier cluster than the use, we will have a liverange
  // overlap and need to add an extra buffer.
  if (schedule.isOpInEarlierCluster(tmemUseStageBoundOps.first,
                                    tmemUseStageBoundOps.second) ||
      (schedule.isOpInSameCluster(tmemUseStageBoundOps.first,
                                  tmemUseStageBoundOps.second) &&
       tmemUseStageBoundOps.first->isBeforeInBlock(
           tmemUseStageBoundOps.second))) {
    tmemUseNumStages += 1;
  }

  // If the accumulator needs to be double-buffered but we can't find the alloc
  // op, then bail out.
  if (tmemUseNumStages > 1 && !alloc.getDefiningOp<ttng::TMEMAllocOp>())
    return forOp;

  OpBuilder builder(forOp);
  Value minusOne =
      arith::ConstantIntOp::create(builder, forOp.getLoc(), -1, 32);
  Value zero = arith::ConstantIntOp::create(builder, forOp.getLoc(), 0, 32);

  // Add arguments to the forOp
  unsigned newOperandIndex = forOp.getInitArgs().size();
  SmallVector<Value> newOperands = {
      zero, // phase
      zero, // barrierIdx
  };
  if (tmemUseNumStages > 1) {
    newOperands.push_back(minusOne); // bufIdx
  }
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;

  int phaseArgIdx = newOperandIndex + 0;
  int barrierIdxArgIdx = newOperandIndex + 1;
  int bufIdxArgIdx = newOperandIndex + 2;
  Value phase = forOp.getRegionIterArg(phaseArgIdx);
  Value barrierIdx = forOp.getRegionIterArg(barrierIdxArgIdx);

  SmallVector<Value> newYieldOperands = {phase, barrierIdx};
  if (tmemUseNumStages > 1) {
    Value bufIdx = forOp.getRegionIterArg(bufIdxArgIdx);
    newYieldOperands.push_back(bufIdx);
  }
  cast<scf::YieldOp>(forOp.getBody()->getTerminator())
      .getResultsMutable()
      .append(newYieldOperands);

  createBarrierAndWaitOps(forOp, schedule, mma, mmaSelfLatency, alloc,
                          phaseArgIdx, barrierIdxArgIdx);

  if (tmemUseNumStages > 1) {
    multibufferTensorMemory(forOp, schedule,
                            alloc.getDefiningOp<ttng::TMEMAllocOp>(),
                            bufIdxArgIdx, tmemUseNumStages);
  }

  return forOp;
}

scf::ForOp lowerMMAs(scf::ForOp forOp, CoarseSchedule &schedule) {
  SmallVector<ttng::MMAv5OpInterface> mmas;
  forOp.walk([&](ttng::MMAv5OpInterface mma) { mmas.push_back(mma); });
  for (auto mma : mmas) {
    forOp = lowerMMA(mma, forOp, schedule);
  }
  return forOp;
}

/////////////////////////////
// LOWER LOOP
/////////////////////////////

void lowerLoop(scf::ForOp forOp,
               triton::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  CoarseSchedule schedule;
  if (failed(schedule.deSerialize(forOp))) {
    return;
  }
  scf::ForOp newForOp = lowerMMAs(forOp, schedule);
  newForOp = lowerLoads(newForOp, schedule, axisInfoAnalysis);
  newForOp = lowerTMADescriptors(newForOp, schedule);
  schedule.serialize(newForOp);
}

} // namespace

void lowerLoops(ModuleOp moduleOp) {
  triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    lowerLoop(forOp, axisInfoAnalysis);
  }
}

} // namespace gpu
} // namespace triton
} // namespace mlir
