#include "mlir/IR/Dominance.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace mlir::triton::gpu {

//===----------------------------------------------------------------------===//
// scheduleLoops
//===----------------------------------------------------------------------===//

template <typename... OpTypes> static bool containsAny(scf::ForOp forOp) {
  WalkResult result = forOp.walk([&](Operation *op) {
    if (isa<OpTypes...>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

// 判断该 for 是否满足软件流水的前置条件。
bool isSafeToPipeline(scf::ForOp forOp) {
  // 跳过 loop-carried 依赖距离 > 1 的循环（expander 暂不支持）。
  if (loopHasDistGreaterThanOne(forOp))
    return false;
  // 不 pipeline 外层循环（体内还嵌套 for/while）。
  if (isOuterLoop(forOp))
    return false;
  // 含 barrier / assert / print 的循环跳过。
  if (containsAny<ttg::BarrierOp, tt::AssertOp, tt::PrintOp>(forOp))
    return false;

  return true;
}

// 处理跨迭代 distance=1 的依赖：
/**
第 i 轮:   addptr → yield %17
                ↓
第 i+1 轮: load 用 %arg6（就是上一轮的 %17）
*/
void scheduleDistanceOneDependencies(scf::ForOp forOp,
                                     CoarseSchedule &schedule) {
  int numStages = schedule.getNumStages();// 3

  // cluster -> 该 cluster 之前新建的 dist1 cluster（按需创建）。
  DenseMap<CoarseSchedule::ClusterHash, CoarseSchedule::Cluster> dist1Cluster;
  /*
  op: loadA/B, convert_layout, dot, addptr, scf.yield
  */
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)//刚进这个函数时，addptr 一般还不在 schedule 里。所以扫到addptr是continue
      continue;
    //扫到已在 schedule 里的 load → 发现操作数是 iter_arg → 从 yield 反查到定义是 addptr → 再 insertIfAbsent(addptr, stage+1, ...)。
    auto [stage, cluster] = schedule[&op];
    // 已在最后一 stage，不能再往后排。
    if (stage == numStages - 1)
      continue;
    //getNestedOperands(&op) 获取当前 op 的所有操作数
    for (Value operand : getNestedOperands(&op)) {//当前已排 op 的所有操作数
      // 操作数是本 block 的 iter_arg（跨迭代依赖）。
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        //操作数是不是 block 参数？%arg6 是 → 进去。普通 %11 那种不是 → 跳过。
        if (arg.getArgNumber() > 0 && arg.getOwner() == op.getBlock()) {
          auto yieldOp = op.getBlock()->getTerminator();//拿到 scf.yield。
          // 这行是在做 iter_arg ↔ yield 操作数 的下标对齐。
          /**
          argNumber 0:  归纳变量 %iv
          argNumber 1:  第 1 个 iter_arg   ← 例如 %arg6 指针
          argNumber 2:  第 2 个 iter_arg   ← %arg7
          argNumber 3:  第 3 个 iter_arg   ← %arg8
          而 scf.yield 的操作数没有 iv，只有要传给下一轮的 carried values：
          yield operand 0 → 下一轮 arg1
          yield operand 1 → 下一轮 arg2
          yield operand 2 → 下一轮 arg3
          所以差 1：
          yield 下标 = arg.getArgNumber() - 1
          */

          /*
          scf.for %iv = %lb to %ub step %step  iter_args(%arg6, %arg7, %arg8)
            %11 = tt.load %arg6 ...          // 当前已排 op，stage=0
            ...
            %17 = tt.addptr %arg6, ...       // 还没排
            scf.yield %17, %18, %16

            %iv 每轮由 for 自己按 step 更新，不经过 yield
            yield 有几个操作数，就对应几个 iter_args
            所以 body 参数是 [iv] + iter_args，而 yield 操作数只有 iter_args，下标才要 - 1
          */
          // yieldOp的参数是3个，arg.getArgNumber()是四个
          //这里每次取出一个operand，上面的%6, %7, %8分三次取出
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          /**
          本轮:  %17 = addptr ...
                scf.yield %17, ...     ← v 是 %17，defOp 是 addptr

          下一轮: %arg6 用的就是这个 %17
          */
          Operation *defOp = v.getDefiningOp();// 找到%17 = addptr ...这个ssa
          if (defOp && schedule.count(defOp) == 0) {//还没有schedule
            if (isa<tt::LoadOp>(defOp)) {// 例外：distance=1 的 load 与当前 op 同排。
              schedule.insertIfAbsent(defOp, stage, cluster);
              schedule.insertDepsOfOp(defOp, stage, cluster,
                                      /*includeArg=*/true,
                                      /*insertIfEarlier=*/true);
            } else {// 普通情况：定义排到下一 stage，cluster 在当前之前。
              // 把当前 op（load）的 cluster 收成一个 hash，当缓存 key。
              /**
              对 GEMM、当前 op 是某个 tt.load（stage=0，cluster 指向 id 为 2 的那个桶）时：
              cluster是load 已有的那个 cluster 迭代器（桶里存的编号一般是 2）。
              不是 2 这个业务编号，而是：reinterpret_cast<size_t>(&*cluster)
              即 list 里那个 int 节点的地址，当成 DenseMap 的 key（一个不透明的整数，如 0x7ffe...）。
              用来区分“是不是同一个 cluster 对象”，不依赖编号之后被 newBefore 改掉。
              */
              CoarseSchedule::ClusterHash clusterHash =
                  CoarseSchedule::hashCluster(cluster);
              //给当前 load 的 cluster 按需建一个“前面的” dist1 桶，并缓存起来。
              /*
              第一次（比如 load A）：map 里没有 → newBefore，得到新 cluster，写进 dist1Cluster
              第二次（load B 同一 cluster）：已有 → 不再建，后面直接用 dist1Cluster[clusterHash] 给 addptr 用
              */
              if (dist1Cluster.count(clusterHash) == 0) {//还没为这个 cluster 建过
                dist1Cluster[clusterHash] =
                    schedule.clusters.newBefore(cluster);//在 load 的 cluster 前面新建一个
              }
              // 把 addptr（defOp）正式写进 schedule
              // defOp ： tt.addptr
              // stage + 1： load 是 0 → addptr 是 1
              // insertIfAbsent：若这个 op 还没有 schedule 记录，就插入；
              // 已经有了就不动（避免两个 load 各插一次把同一个 addptr 改乱）。
              schedule.insertIfAbsent(defOp, stage + 1,
                                      dist1Cluster[clusterHash]);
              // 看addptr的操作数，GEMM会直接跳过，do nothing
              schedule.insertDepsOfOp(defOp, stage + 1,
                                      dist1Cluster[clusterHash],
                                      /*includeArg=*/true,
                                      /*includeIfEarlier=*/true);
            }
          }
        }
      }
    }
  }
}

// 把仍未入 schedule 的 op 全部放到最后一 stage。
// 同时保证：use 的 cluster 不能早于其 def（必要时把 user 往后推）。
void scheduleRemainingToLastStage(scf::ForOp forOp, CoarseSchedule &schedule,
                                  CoarseSchedule::Cluster afterPrologue) {
  int numStages = schedule.getNumStages();
  // 未排 op 先默认放到 afterPrologue 这个 cluster。
  DenseMap<Operation *, CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0) {
      opToCluster[&op] = afterPrologue;
    }
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // 只从最后一 stage 的生产者出发传播 cluster 约束。
    if (stage == numStages - 1) {
      queue.push_back(op);
    }
  }
  // 若 user 的 cluster 早于 def，则把 user 推到与 def 同 cluster。
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        CoarseSchedule::Cluster userCluster = opToCluster[user];
        CoarseSchedule::Cluster opCluster;
        if (schedule.count(op))
          opCluster = schedule[op].second;
        else
          opCluster = opToCluster[op];
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, numStages - 1, cluster);
  }
}

namespace {
// 循环体内是否已有任意 op 带 latency（在 opLatency map 里）。
bool hasLatenciesAssigned(scf::ForOp forOp,
                          const DenseMap<Operation *, int> &opLatency) {
  // 循环体里 load、dot、addptr 那些，不包括结尾的 scf.yield
  /*
  scf.for                    ← 这是一个 Operation
  └── Region（循环体）
        └── Block
              ├── tt.load      ← 里面的 Operation
              ├── tt.dot
              ├── tt.addptr
              └── scf.yield    ← terminator
  */
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opLatency.count(&op))// 这里是在找这个forop中是否至少有一个op出现在map中
      return true;
  }
  return false;
}

// 根据 latency 给“关键路径”上的 op 分配初始 stage/cluster。
// 核心：从 latency op 出发算到 yield 的最长路径距离，再反推 stage。
CoarseSchedule scheduleKeyOps(scf::ForOp forOp,
                              const DenseMap<Operation *, int> &opLatency) {
  llvm::MapVector<Operation *, int> opToStage;// 用于生成各个op的stage
  auto terminator = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  // 收集当前forOp所有带 latency 的 op（通常是可异步的 load 等）。opLatency是整个module的map
  SmallVector<Operation *> latOps;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opLatency.count(&op))
      latOps.push_back(&op);
  }
  if (latOps.empty())
    return CoarseSchedule(0);

  DominanceInfo domInfo(forOp);//可删除，没用上

  // distance[op] = 自身 latency + 下游 user 的最大 distance（到 yield 的最长路）。
  DenseMap<Operation *, int> distance;
  // 沿着 use 链朝 yield 方向，找加权最长路；yield 本身不参与，只是路径尽头。
  std::function<int(Operation *)> computeDistance = [&](Operation *op) -> int {
    auto it = distance.find(op);
    if (it != distance.end())
      return it->second;
    int maxDist = -1;
    for (Operation *user : op->getUsers()) {
      //第一轮：user是convert_layout,在forOp中，所以直接返回convert_layout本身
      //第二轮：user是dot,在forOp中，所以直接返回dot本身
      //第三轮：user是scf.yield,在forOp中，所以直接返回scf.yield本身
      Operation *inBlockUser = forOp.getBody()->findAncestorOpInBlock(*user);
      if (!inBlockUser || inBlockUser == terminator)
        continue;
      int distUser = computeDistance(inBlockUser);
      if (distUser > maxDist)
        /*
        distUser = computeDistance(dot) = 0, maxDist = 0;
        distUser = computeDistance(convert_layout) = 0, maxDist = 0;
        distUser = computeDistance(loadA/B) = 2, maxDist = 0;
        */
        maxDist = distUser;
    }
    int lat = 0;
    if (opLatency.count(op))
      lat = opLatency.lookup(op);
    // 无 user 时：若自身有 latency 仍计入，否则贡献 0。
    int d = lat + (maxDist < 0 ? 0 : maxDist);
    /* 
    distance[dot] = 0;
    distance[convert_layout] = 0;
    distance[loadA/B] = 2;
    */
    distance[op] = d;
    return d;
  };

  // 从所有 latency op 出发，取全局最大距离。
  int maxDistance = 0;
  for (Operation *latOp : latOps) {
    // 只有带latency的load才走computeDistance，addptr不走这里
    int d = computeDistance(latOp);
    if (d > maxDistance)
      maxDistance = d;
  }

  // stage = maxDistance - dist：距离 yield 越远（越大 dist）→ 越早的 stage。
  // 例：load latency=2、dot distance=0 → load 在 stage 0，dot 在 stage 2。
  for (auto [op, dist] : distance) {
    if (dist >= 0)
      opToStage[op] = maxDistance - dist;
  }

  // 接下来把stage变成cluster
  auto stages = llvm::make_second_range(opToStage); // 取出opToStage的second，即stage
  int maxStage = *llvm::max_element(stages); // 取出stages中的最大值，为2
  CoarseSchedule schedule(maxStage + 1);
  SmallVector<CoarseSchedule::Cluster> clusters(maxStage + 1); // 创建maxStage + 1个cluster
  for (int i = 0; i <= maxStage; i++) {//
    clusters[i] = schedule.clusters.newAtBack();
  }
  // 按 stage 从高到低插入，使 cluster 内大致呈反向程序序。
  for (auto [op, stage] : opToStage)
    schedule.insert(op, stage, clusters[maxStage - stage]);

  // 把已排路径上的 scf.if（且自身不是 latency op）尽量推到 epilogue cluster。
  CoarseSchedule::Cluster epilogue = schedule.clusters.newAtBack();
  for (auto [op, stage] : opToStage) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp)
      continue;
    if (opLatency.contains(ifOp))
      continue;
    // 若 if 的 forward slice 里已有已排 op，再推后会造成依赖冲突，跳过。
    SetVector<Operation *> slice;
    getForwardSlice(ifOp, &slice);
    if (llvm::any_of(slice, [&](Operation *op) { return opToStage.count(op); }))
      continue;
    schedule.insert(ifOp, stage, epilogue);
  }

  return schedule;
}

// 为单个 for 生成初始 schedule，后续步骤在此基础上补依赖。
CoarseSchedule getInitialSchedule(scf::ForOp forOp,
                                  const DenseMap<Operation *, int> &opLatency) {
  // 不满足安全条件 → 空 schedule（调用方会直接 return）。
  if (!isSafeToPipeline(forOp))
    return CoarseSchedule(0);

  // 常见路径：已有 assign-latencies 结果 → 按 latency 排关键 op。
  if (hasLatenciesAssigned(forOp, opLatency))
    // 这里直排关键op的stage和cluster，即按照DFS load -> convert_layout -> dot ，不包括addptr
    return scheduleKeyOps(forOp, opLatency);

  // Warp-specialize 路径：循环上已有 stage/cluster，反序列化后复用/裁剪。
  CoarseSchedule schedule;
  if (forOp->hasAttr(kWarpSpecializeAttrName) &&
      succeeded(schedule.deSerialize(forOp))) {
    // 该 for 来自 warp-specialize 分区，可能只看到原循环部分 stage。
    auto isLatencyOp = [&](Operation &op) {
      return opLatency.count(&op) ||
             isa<LoadOp, DescriptorLoadLikeOpInterface, LocalStoreOp,
                 LocalLoadOp, ttng::TMEMLoadOp, ttng::TMEMStoreOp,
                 AsyncCopyGlobalToLocalOp, ttng::TMAOpInterface,
                 ttng::MMAv5OpInterface, ttng::WaitBarrierOp,
                 ttng::ArriveBarrierOp>(op);
    };

    // latency 相关 op 都在同一 stage（或没有）→ 不需要多 stage，压成 1 stage。
    DenseSet<int> latencyStages;
    auto ops = forOp.getBody()->without_terminator();
    for (Operation &op : llvm::make_filter_range(ops, isLatencyOp)) {
      // FIXME: This should assert all latency ops have an assigned stage.
      if (schedule.count(&op))
        latencyStages.insert(schedule[&op].first);
    }
    if (latencyStages.size() <= 1) {
      CoarseSchedule normalized(/*numStages=*/1);
      auto cluster = normalized.clusters.newAtFront();
      for (Operation &op : ops)
        normalized.insert(&op, 0, cluster);
      return normalized;
    }

    // 多 stage：裁掉用不到的 stage 后返回。
    schedule.shrinkToFit();
    return schedule;
  }

  // 既无 latency 也无已有 schedule → 空。
  return CoarseSchedule(0);
}

// 把循环内 scf.if 尽量推到边界：相关 if → prologue，其余 → epilogue。
// 返回 prologue 之后的 cluster（无 prologue 则为 clusters.begin()）。
CoarseSchedule::Cluster schedulePrologueAndEpilogue(scf::ForOp forOp,
                                                    CoarseSchedule &schedule) {
  int numStages = schedule.getNumStages();
  CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  // 找落在“已排 op 反切片”上的 if，记为 prologue 候选。
  DenseMap<scf::IfOp, int> ifsToStage;
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : schedule.getOpsInOrder(forOp)) {
      if (stage_ != stage)
        continue;
      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      opt.omitUsesFromAbove = false;
      (void)getBackwardSlice((Operation *)op, &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  // prologue：新建最前 cluster，放入相关 if。
  if (!ifsToStage.empty()) {
    CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
    for (auto [ifOp, stage] : ifsToStage) {
      schedule.insertIfAbsent(ifOp, stage, prologueCluster);
    }
  }

  // 其余 if 推到最后一 stage 的 epilogue cluster。
  CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (ifsToStage.count(ifOp) == 0) {
        schedule.insertIfAbsent(ifOp, numStages - 1,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

void scheduleLoop(scf::ForOp forOp,
                  const DenseMap<Operation *, int> &opLatency) {
  // 1) 根据 latency（或已有 schedule）生成初始粗粒度 schedule：
  //    给关键 op（如带 tt.latency 的 load）分配 stage/cluster。
  //    不安全或没有可排内容时返回空 schedule。
  CoarseSchedule schedule = getInitialSchedule(forOp, opLatency);
  if (schedule.empty())
    return;
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Initial coarse schedule:\n" << forOp << "\n";
  });

  // 2) GEMM会跳过这里
  // 处理循环里的 scf.if：
  //    - 落在已排 op 反切片上的 if → 尽量推到循环开头（prologue）
  //    - 其余 if → 推到循环末尾（epilogue）
  //    返回值 afterPrologue：prologue 之后的 cluster，后面给“剩余 op”用。
  CoarseSchedule::Cluster afterPrologue =
      schedulePrologueAndEpilogue(forOp, schedule);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Coarse schedule with prologue and epilogue:\n" << forOp << "\n";
  });

  // 3) GEMM 会跳过这里
  // 把已排 op 的依赖（定义链上的未排 op）拉进同一 stage/cluster，例如arith.cmpi计算k<K时
  scheduleDependencies(forOp, schedule);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Coarse schedule with dependencies:\n" << forOp << "\n";
  });

  // 4) 处理跨迭代 distance=1 的依赖：
  //    本轮用到的是上一轮 yield 过来的值，定义要排到下一 stage
  //    （load 的 distance=1 例外，可与当前 op 同 stage）。
  // GEMM在这里处理addptr这个op
  scheduleDistanceOneDependencies(forOp, schedule);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Coarse schedule with dist 1:\n" << forOp << "\n";
  });

  // 5) GEMM 会跳过这里
  // 仍未入 schedule 的 op，放到最后一 stage（cluster 不低于 afterPrologue）。
  scheduleRemainingToLastStage(forOp, schedule, afterPrologue);
  LLVM_DEBUG({
    schedule.serialize(forOp);
    DBGS() << "Final coarse schedule:\n" << forOp << "\n";
  });

  // 6) 写回 IR：各 op 的 loop.stage / loop.cluster，
  //    以及 for 上的 tt.scheduled_max_stage。
  schedule.serialize(forOp);
}

/// Schedule the loops based on the latencies assigned to the operations.
void scheduleLoops(ModuleOp moduleOp) {
  // 首先把latency attr提取出来，然后删掉IR中的latency attr
  DenseMap<Operation *, int> opLatency = deserializeLatencies(moduleOp);
  SmallVector<scf::ForOp> loops;
  // 从整个 module 里把所有 scf.for 收集进 loops
  // 深度优先遍历 module 下所有 Operation, 回调参数写成 scf::ForOp：只在遇到 for 时触发
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    // schedule每一个scf::ForOp
    scheduleLoop(forOp, opLatency);
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_TRITONGPUSCHEDULELOOPS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct ScheduleLoops : public impl::TritonGPUScheduleLoopsBase<ScheduleLoops> {
  using TritonGPUScheduleLoopsBase::TritonGPUScheduleLoopsBase;

  void runOnOperation() override { scheduleLoops(getOperation()); }
};

} // namespace mlir::triton::gpu
