//===- LoopPipelining.cpp - Code to perform loop software pipelining-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop software pipelining
//
//===----------------------------------------------------------------------===//

// Fork of upstream pipeliner. This will be merged upstream once things are
// stable. Modifications so far are:
// -Bug fix for def with a distance of 1 scheduled in stage 0.
// -Support dynamic loops and predicate operations in the prologue.
// -Support for non-index type for induction variable.
// -Support source with distance of 1 used multiple stages later.
// -Fix bug when a value yield is used outside the loop and the value def is not
// in the last stage. If we are not peeling the epilgue we need to remap the
// output correctly.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"

// FIXME: PipelineExpander should not depend on Triton-specific headers!
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#define DEBUG_TYPE "triton-loop-pipelining"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::scf;
using namespace mlir::triton;

namespace {

/// Helper to keep internal information during pipelining transformation.
struct LoopPipelinerInternal {
  /// Coarse liverange information for ops used across stages.
  struct LiverangeInfo {
    unsigned lastUseStage = 0;
    unsigned defStage = 0;
  };

protected:
  ForOp forOp;
  unsigned maxStage = 0;
  DenseMap<Operation *, unsigned> stages;
  std::vector<Operation *> opOrder;
  Value ub;
  Value lb;
  Value step;
  bool dynamicLoop;
  triton::PipeliningOption::AnnotationlFnType annotateFn = nullptr;
  bool peelEpilogue;
  triton::PipeliningOption::PredicateOpFnType predicateFn = nullptr;
  triton::PipeliningOption::EmitPredicateStageFnType emitPredicateStageFn =
      nullptr;

  // When peeling the kernel we generate several version of each value for
  // different stage of the prologue. This map tracks the mapping between
  // original Values in the loop and the different versions
  // peeled from the loop.
  DenseMap<Value, llvm::SmallVector<Value>> valueMapping;

  /// Assign a value to `valueMapping`, this means `val` represents the version
  /// `idx` of `key` in the epilogue.
  void setValueMapping(Value key, Value el, int64_t idx);

  /// Return the defining op of the given value, if the Value is an argument of
  /// the loop return the associated defining op in the loop and its distance to
  /// the Value.
  std::pair<Operation *, int64_t> getDefiningOpAndDistance(Value value);

  /// Return true if the schedule is possible and return false otherwise. A
  /// schedule is correct if all definitions are scheduled before uses.
  bool verifySchedule();

public:
  /// Initialize the information for the given `op`, return true if it
  /// satisfies the pre-condition to apply pipelining.
  bool initializeLoopInfo(ForOp op, const triton::PipeliningOption &options);
  /// Emits the prologue, this creates `maxStage - 1` part which will contain
  /// operations from stages [0; i], where i is the part index.
  LogicalResult emitPrologue(RewriterBase &rewriter);
  /// Gather liverange information for Values that are used in a different stage
  /// than its definition.
  llvm::MapVector<Value, LiverangeInfo> analyzeCrossStageValues();
  scf::ForOp createKernelLoop(
      const llvm::MapVector<Value, LiverangeInfo> &crossStageValues,
      RewriterBase &rewriter,
      llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap);
  /// Emits the pipelined kernel. This clones loop operations following user
  /// order and remaps operands defined in a different stage as their use.
  LogicalResult createKernel(
      scf::ForOp newForOp,
      const llvm::MapVector<Value, LiverangeInfo> &crossStageValues,
      const llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap,
      RewriterBase &rewriter);
  /// Emits the epilogue, this creates `maxStage - 1` part which will contain
  /// operations from stages [i; maxStage], where i is the part index.
  LogicalResult emitEpilogue(RewriterBase &rewriter,
                             llvm::SmallVector<Value> &returnValues);
};

/// Find operands of all the nested operations within `op`.
static SetVector<Value> getNestedOperands(Operation *op) {
  SetVector<Value> operands;
  op->walk([&](Operation *nestedOp) {
    for (Value operand : nestedOp->getOperands()) {
      operands.insert(operand);
    }
  });
  return operands;
}

bool LoopPipelinerInternal::initializeLoopInfo(
    ForOp op, const triton::PipeliningOption &options) {
  LDBG("Start initializeLoopInfo");
  forOp = op;
  ub = forOp.getUpperBound();//上界（结束，如 K 到哪结束）
  lb = forOp.getLowerBound();//下界（起始，如 K 从哪开始）
  step = forOp.getStep();//步长（一块 K-tile 多大）

  std::vector<std::pair<Operation *, unsigned>> schedule;
  options.getScheduleFn(forOp, schedule);
  if (schedule.empty()) {
    LDBG("--empty schedule -> BAIL");
    return false;
  }

  /*
  把 getScheduleFn 拿到的 finalSchedule（{op, stage} 列表）拆进三个内部结构：
  stages[op] = stage ：每个 op 的 stage（查表用）
  opOrder： op 的顺序列表（clone 进 prologue/kernel 时按这个序）
  maxStage： 最大 stage 号（3 stage 时一般为 2）
  */
  opOrder.reserve(schedule.size());
  for (auto &opSchedule : schedule) {
    maxStage = std::max(maxStage, opSchedule.second);
    stages[opSchedule.first] = opSchedule.second;
    opOrder.push_back(opSchedule.first);
  }

  dynamicLoop = true;
  //在问：循环的 ub/lb/step 是不是编译期常量。
  auto upperBoundCst = ub.getDefiningOp<arith::ConstantIndexOp>();
  auto lowerBoundCst = lb.getDefiningOp<arith::ConstantIndexOp>();
  auto stepCst = step.getDefiningOp<arith::ConstantIndexOp>();
  if (!upperBoundCst || !lowerBoundCst || !stepCst) {
    if (!options.supportDynamicLoops) {
      LDBG("--dynamic loop not supported -> BAIL");
      return false;
    }
  } else {//在 lb/ub/step 都是常量 的前提下，已经算出静态迭代次数 numIteration
    int64_t ubImm = upperBoundCst.value();
    int64_t lbImm = lowerBoundCst.value();
    int64_t stepImm = stepCst.value();
    int64_t numIteration = llvm::divideCeilSigned(ubImm - lbImm, stepImm);
    if (numIteration >= maxStage) {
      dynamicLoop = false;
    } else if (!options.supportDynamicLoops) {
      LDBG("--fewer loop iterations than pipeline stages -> BAIL");
      return false;
    }
  }
  peelEpilogue = options.peelEpilogue;
  predicateFn = options.predicateFn;
  if ((!peelEpilogue || dynamicLoop) && predicateFn == nullptr) {
    LDBG("--no epilogue or predicate set -> BAIL");
    return false;
  }
  emitPredicateStageFn = options.emitPredicateStageFn;
  if (emitPredicateStageFn == nullptr) {
    emitPredicateStageFn = mlir::triton::emitPredicateForStage;
  }

  // All operations need to have a stage.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!stages.contains(&op)) {
      op.emitOpError("not assigned a pipeline stage");
      LDBG("--op not assigned a pipeline stage: " << op << " -> BAIL");
      return false;
    }
  }

  //检查：按当前 stage + opOrder 展开后，会不会出现「消费者比生产者还早执行」。
  if (!verifySchedule()) {
    LDBG("--invalid schedule: " << op << " -> BAIL");
    return false;
  }

  // Currently, we do not support assigning stages to ops in nested regions. The
  // block of all operations assigned a stage should be the single `scf.for`
  // body block.
  for (const auto &[op, stageNum] : stages) {
    (void)stageNum;
    if (op == forOp.getBody()->getTerminator()) {
      op->emitError("terminator should not be assigned a stage");
      LDBG("--terminator should not be assigned stage: " << *op << " -> BAIL");
      return false;
    }
    if (op->getBlock() != forOp.getBody()) {
      op->emitOpError("the owning Block of all operations assigned a stage "
                      "should be the loop body block");
      LDBG("--the owning Block of all operations assigned a stage "
           "should be the loop body block: "
           << *op << " -> BAIL");
      return false;
    }
  }

  // Support only loop-carried dependencies with a distance of one iteration or
  // those defined outside of the loop. This means that any dependency within a
  // loop should either be on the immediately preceding iteration, the current
  // iteration, or on variables whose values are set before entering the loop.
  for (auto &op : forOp.getBody()->without_terminator()) {
    for (auto operand : getNestedOperands(&op)) {
      auto [def, distance] = getDefiningOpAndDistance(operand);
      if (!def)
        continue;
      if (distance > 1) {
        LDBG("--only support loop carried dependency with a distance of 1 or "
             "defined outside of the loop -> BAIL");
        return false;
      }
    }
  }
  annotateFn = options.annotateFn;
  return true;
}

/// Compute unrolled cycles of each op (consumer) and verify that each op is
/// scheduled after its operands (producers) while adjusting for the distance
/// between producer and consumer.
bool LoopPipelinerInternal::verifySchedule() {
  int64_t numCylesPerIter = opOrder.size();
  // Pre-compute the unrolled cycle of each op.
  DenseMap<Operation *, int64_t> unrolledCyles;
  for (int64_t cycle = 0; cycle < numCylesPerIter; cycle++) {
    Operation *def = opOrder[cycle];
    auto it = stages.find(def);
    assert(it != stages.end());
    int64_t stage = it->second;
    unrolledCyles[def] = cycle + stage * numCylesPerIter;
  }
  for (Operation *consumer : opOrder) {
    int64_t consumerCycle = unrolledCyles[consumer];
    for (Value operand : getNestedOperands(consumer)) {
      auto [producer, distance] = getDefiningOpAndDistance(operand);
      if (!producer)
        continue;
      auto it = unrolledCyles.find(producer);
      // Skip producer coming from outside the loop.
      if (it == unrolledCyles.end())
        continue;
      int64_t producerCycle = it->second;
      if (consumerCycle < producerCycle - numCylesPerIter * distance) {
        InFlightDiagnostic diag =
            consumer->emitWarning("operation scheduled before its operands. "
                                  "Pipelining will be disabled.");
        diag.attachNote(producer->getLoc())
            .append("operand defined here: ")
            .appendOp(*producer, OpPrintingFlags().printGenericOpForm());
        return false;
      }
    }
  }
  return true;
}

/// Clone `op` and call `callback` on the cloned op's operands as well as any
/// operands of nested ops that:
/// 1) aren't defined within the new op or
/// 2) are block arguments.
static Operation *
cloneAndUpdateOperands(RewriterBase &rewriter, Operation *op,
                       function_ref<void(OpOperand *newOperand)> callback) {
  // clone一个op
  Operation *clone = rewriter.clone(*op);
  /*
  第一次，lambda的参数nested就是clone后的op
  若这条 op 带 region（如 scf.if），walk 还会继续进内部，后面的 nested 是 region 里嵌套的子 op。
  对 GEMM 里常见的扁平 async_copy / commit：没有内嵌 region，整个 walk 往往只有一次，nested == clone。
  */
  clone->walk<WalkOrder::PreOrder>([&](Operation *nested) {
    // 'clone' itself will be visited first.
    for (OpOperand &operand : nested->getOpOperands()) {
      /*
      operand ：某个 OpOperand（「这个 op 的第几个输入槽」）。
                比如 GEMM 的 async_copy，有 4 个输入槽：%ptrA, %slot0, %mask, %other。
                operand 就是 %ptrA, %slot0, %mask, %other 中的一个。
                operand.get() 就是 %ptrA, %slot0, %mask, %other 中的一个 Value。
     getDefiningOp() ：找到对应的SSA value定义的地方。返回 Operation*。若无，返回 nullptr。
     例如：async_copy 的指针操作数若来自 %p = tt.addptr ...，则 getDefiningOp() 是那条 addptr；
          若来自 for 的 %arg6，则是 nullptr，后面靠 isa<BlockArgument> 识别。
      */
      Operation *def = operand.get().getDefiningOp();
      /*
      判断clone是否是def的祖先，也就是op是否是oprand的祖先
      clone 刚复制完时，操作数仍指向 旧 for / 外面 的 op，此时为false，此时交给callback做valueMapping替换
      操作数由 clone 内部 新定义的子 op 提供，此时为true
      */
      if ((def && !clone->isAncestor(def)) || isa<BlockArgument>(operand.get()))
        callback(&operand);
    }
  });
  return clone;
}

LogicalResult LoopPipelinerInternal::emitPrologue(RewriterBase &rewriter) {
  // Initialize the iteration argument to the loop initiale values.
  /*
  以 lower 之后、expand 之前 的 GEMM for 为例（概念上）：
  scf.for %iv = %lb to %ub step %step
      iter_args(
        %ptrA    = %initA,    // 初值：第一块 A 指针
        %ptrB    = %initB,
        %acc     = %cst0,     // 累加器 0
        %insert  = %c-1,
        %extract = %c0
      ) {
    // body 里用 %ptrA, %acc, ...
    %copy = async_copy %ptrA, ...
    ...
    scf.yield %nextA, %nextB, %newAcc, %nextIns, %nextExt
  }
  下面的这两行代码建立起：key（body 里的名字）-> value（iter_args 的初值）的 mapping
  %ptrA -> %initA
  %ptrB -> %initB
  %acc -> %cst0
  %insert -> %c-1
  %extract -> %c0
  为什么需要这个 mapping？
  Prologue 要在 for 外面 clone 第 0 轮的 async_copy。
  Body 里写的是 %ptrA，for 外没有这个 block arg，所以 clone 时查表：
      version=0 的 %ptrA → 用 %initA 替换
  于是 prologue 里变成：
  // 相当于 S0(0)
  %copy0 = async_copy %initA, %slot0, ...
  而不是非法的 async_copy %ptrA。
  setValueMapping 本身:
  valueMapping[key][idx] = el：同一个 SSA 概念值可以有多版（prologue 第 0、1 段…）。这里只初始化 第 0 版 = 循环 init。
  这里只是记录一个mapping，方便后面clone使用
  */
  for (auto [arg, operand] :
       llvm::zip(forOp.getRegionIterArgs(), forOp.getInitsMutable())) {
    setValueMapping(arg, operand.get(), 0);
  }

  // If the incoming value to an iter arg from the loop yield is defined outside
  // the loop, then that means the iter arg takes that value for all stages
  // after the first stage.
  //这段GEMM不会执行
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (auto [arg, operand] :
       llvm::zip(forOp.getRegionIterArgs(), yield->getOpOperands())) {
    if (forOp.getBodyRegion().isAncestor(operand.get().getParentRegion()))
      continue;
    for (int64_t i = 1; i < maxStage; ++i)
      setValueMapping(arg, operand.get(), i);
  }

  Location loc = forOp.getLoc();
  SmallVector<Value> predicates(maxStage);
  for (int64_t i = 0; i < maxStage; i++) {
    // special handling for induction variable as the increment is implicit.
    // iv = lb + i * step
    Type t = lb.getType();
    Value iv = arith::AddIOp::create(
        rewriter, loc, lb,
        arith::MulIOp::create(
            rewriter, loc, step,
            arith::ConstantOp::create(rewriter, loc,
                                      rewriter.getIntegerAttr(t, i))));
    /*
    建立 induction variable 的 mapping， 
    valueMapping[ 原for的inductionVar ][ 版本号 i ] = 刚算好的 iv表达式
    */
    setValueMapping(forOp.getInductionVar(), iv, i);

    /*
    arith.cmpi，谓词是 slt（signed less than）：
    %pred_i = arith.cmpi slt, %iv, %ub : index
    */
    if (dynamicLoop) {
      // pred = ub > lb + (i * step)
      predicates[i] = arith::CmpIOp::create(rewriter, loc,
                                            arith::CmpIPredicate::slt, iv, ub);
    }
    /*
    opOrder = [
      convert_A, convert_B, mulf, dot,   // stage 2, cluster 0
      addptr_A, addptr_B,                // stage 1, cluster 2
      load_A, load_B                     // stage 0, cluster 3
    ]
    cluster可以排出dot addptr load指令在一个forloop中的先后顺序
    */

    /*
    只用 A 的指针 把 这个for循环走一遍

    原 body 里相关的两件事：
    scf.for ... iter_args(%ptrA = %initA, ...) {
      // stage 0
      %tok = async_copy %ptrA, ...

      // stage 1
      %nextA = addptr %ptrA, 4

      scf.yield %nextA, ...   // %nextA 交给下一轮的 %ptrA
    }
    yield 的第 0 个操作数是 %nextA → 对应 iter_arg %ptrA。

    时刻 0：只 clone stage 0（copy）
    开头表：%ptrA[0] = %initA
    clone async_copy：
    1.查表用 %ptrA[0] → %initA
    2.copy 的结果是 token，不是 yield 给 %ptrA 的那个值
    3.内层 if (operand.get() != op->getResult(...)) 对 %ptrA 对不上 → 531 这行对 %ptrA 不执行
    表不变：%ptrA[0] 仍是 %initA。
    lp3 对应：第一次 async_copy %4, 槽0。

    时刻 1：先 clone addptr（stage 1）——531行 在这里开火
    i=1，stages[addptr]=1。
    ① clone
    读操作数：%ptrA[i-stage] = %ptrA[0] = %initA
    得到：%23 = addptr %initA, 4
    ② 发现 %nextA 被 yield
    operand.get() == addptr 的结果   // 命中
    → setValueMapping(%ptrA, %23, i - stages[op] + 1)
    //                              1 - 1 + 1 = 1
    表变成：
    %ptrA[0] = %initA
    %ptrA[1] = %23      ← 就是 531–534 写进去的
    含义：原循环里「下一轮的 %ptrA」= %nextA；prologue 里这个下一版就是 %23。

    时刻 1：再 clone 第二次 async_copy（stage 0）
    读：%ptrA[i-stage] = %ptrA[1-0] = %ptrA[1] = %23
    ⇒ async_copy %23, 槽1
    lp3 里第二次 copy 用的就是 %23。
    若没有 531–534： %ptrA[1] 空着，第二次 copy 还是用 %initA，两块都拷同一地址 → 错。

    对照代码三个参数：
    setValueMapping(
      forOp.getRegionIterArgs()[...],  // key = %ptrA（iter_arg，不是 %nextA）
      source,                          // el  = %23（clone 出的 addptr 结果）
      i - stages[op] + 1);             // idx = 1（下一版）
    --------------------------------
    key：原 for 里的 iter_arg %ptrA
    el：clone 出的 addptr 结果 %23
    idx：下一版下标 1

    时间线小结
    init:        %ptrA[0]=%initA

    i=0 copy:    用 [0]，不写 %ptrA

    i=1 addptr:  用 [0] 算出 %23
                466行：%ptrA[1]=%23     ← 你问的这句

    i=1 copy:    用 [1] 即 %23
    */
    for (Operation *op : opOrder) {
      /*
      若 op 的 stage 大于 i，则跳过（因为 op 还没执行到）。
      */
      if (stages[op] > i)
        continue;
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            auto it = valueMapping.find(newOperand->get());
            if (it != valueMapping.end()) {
              Value replacement = it->second[i - stages[op]];
              newOperand->set(replacement);
            }
          });
      //这里是处理越界的corner case，可以先跳过这里
      int predicateIdx = i - stages[op];
      if (predicates[predicateIdx]) {
        OpBuilder::InsertionGuard insertGuard(rewriter);
        newOp = predicateFn(rewriter, newOp, predicates[predicateIdx]);
        if (newOp == nullptr)
          return failure();
      }
      if (annotateFn)//GEMM跳过这里
        annotateFn(newOp, triton::PipeliningOption::PipelinerPart::Prologue, i);
      /*
      遍历这条 op 的每个结果：
      addptr → 1 个结果（新指针）
      async_copy → 1 个 token（作为下一步的 waitToken）
      dot → 1 个 acc（prologue 里通常 stage 不够，不会走到这里）
      */
      for (unsigned destId : llvm::seq(unsigned(0), op->getNumResults())) {
        Value source = newOp->getResult(destId);
        // If the value is a loop carried dependency update the loop argument
        /*
        原 body 里，这个 op 的结果有没有被 scf.yield 交给下一轮 iter_arg
        addptr 会命中：
        %nextA = tt.addptr %ptrA, 4      // op 的 result
        scf.yield %nextA, ...            // yield 带走 → 对应 iter_arg %ptrA
        命中后做setValueMapping(%ptrA,  i - stages[addptr] + 1);
        把「yield 出去的那个新值」记成对应 iter_arg 的下一版。
        */
        for (OpOperand &operand : yield->getOpOperands()) {
          if (operand.get() != op->getResult(destId))
            continue;
          if (predicates[predicateIdx] &&
              !forOp.getResult(operand.getOperandNumber()).use_empty()) {
            // If the value is used outside the loop, we need to make sure we
            // return the correct version of it.
            Value prevValue = valueMapping
                [forOp.getRegionIterArgs()[operand.getOperandNumber()]]
                [i - stages[op]];
            source = arith::SelectOp::create(
                rewriter, loc, predicates[predicateIdx], source, prevValue);
          }
          setValueMapping(forOp.getRegionIterArgs()[operand.getOperandNumber()],// 如 %ptrA
                          source,// clone 出的新值，如 %23
                           i - stages[op] + 1);//下一版下标
        }
        setValueMapping(op->getResult(destId), newOp->getResult(destId),
                        i - stages[op]);
      }
    }
  }
  return success();
}

llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
LoopPipelinerInternal::analyzeCrossStageValues() {
  llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo> crossStageValues;
  for (Operation *op : opOrder) {
    unsigned stage = stages[op];

    auto analyzeOperand = [&](OpOperand &operand) {
      auto [def, distance] = getDefiningOpAndDistance(operand.get());
      if (!def)
        return;
      auto defStage = stages.find(def);
      if (defStage == stages.end() || defStage->second == stage ||
          defStage->second == stage + distance)
        return;
      assert(stage > defStage->second);
      LiverangeInfo &info = crossStageValues[operand.get()];
      info.defStage = defStage->second;
      info.lastUseStage = std::max(info.lastUseStage, stage);
    };

    for (OpOperand &operand : op->getOpOperands())
      analyzeOperand(operand);
    visitUsedValuesDefinedAbove(op->getRegions(), [&](OpOperand *operand) {
      analyzeOperand(*operand);
    });
  }
  return crossStageValues;
}

std::pair<Operation *, int64_t>
LoopPipelinerInternal::getDefiningOpAndDistance(Value value) {
  return triton::getDefiningOpAndDistance(forOp, value);
}

scf::ForOp LoopPipelinerInternal::createKernelLoop(
    const llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
        &crossStageValues,
    RewriterBase &rewriter,
    llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap) {
  // Creates the list of initial values associated to values used across
  // stages. The initial values come from the prologue created above.
  // Keep track of the kernel argument associated to each version of the
  // values passed to the kernel.
  llvm::SmallVector<Value> newLoopArg;
  // For existing loop argument initialize them with the right version from the
  // prologue.
  for (const auto &retVal :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
    Operation *def = retVal.value().getDefiningOp();
    auto defStage = stages.find(def);
    if (defStage != stages.end()) {
      Value valueVersion =
          valueMapping[forOp.getRegionIterArgs()[retVal.index()]]
                      [maxStage - defStage->second];
      assert(valueVersion);
      newLoopArg.push_back(valueVersion);
    } else
      newLoopArg.push_back(forOp.getInitArgs()[retVal.index()]);
  }
  for (auto escape : crossStageValues) {
    LiverangeInfo &info = escape.second;
    Value value = escape.first;
    for (unsigned stageIdx = 0; stageIdx < info.lastUseStage - info.defStage;
         stageIdx++) {
      Value valueVersion =
          valueMapping[value][maxStage - info.lastUseStage + stageIdx];
      assert(valueVersion);
      newLoopArg.push_back(valueVersion);
      loopArgMap[std::make_pair(value, info.lastUseStage - info.defStage -
                                           stageIdx)] = newLoopArg.size() - 1;
    }
  }

  // Create the new kernel loop. When we peel the epilgue we need to peel
  // `numStages - 1` iterations. Then we adjust the upper bound to remove those
  // iterations.
  Value newUb = forOp.getUpperBound();
  if (peelEpilogue) {
    Type t = ub.getType();
    Location loc = forOp.getLoc();
    // newUb = ub - maxStage * step
    Value maxStageValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(t, maxStage));
    Value maxStageByStep =
        arith::MulIOp::create(rewriter, loc, step, maxStageValue);
    newUb = arith::SubIOp::create(rewriter, loc, ub, maxStageByStep);
  }
  auto newForOp =
      scf::ForOp::create(rewriter, forOp.getLoc(), forOp.getLowerBound(), newUb,
                         forOp.getStep(), newLoopArg);
  newForOp->setAttrs(forOp->getAttrs());
  // When there are no iter args, the loop body terminator will be created.
  // Since we always create it below, remove the terminator if it was created.
  if (!newForOp.getBody()->empty())
    rewriter.eraseOp(newForOp.getBody()->getTerminator());
  return newForOp;
}

LogicalResult LoopPipelinerInternal::createKernel(
    scf::ForOp newForOp,
    const llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
        &crossStageValues,
    const llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &loopArgMap,
    RewriterBase &rewriter) {
  valueMapping.clear();

  // Create the kernel, we clone instruction based on the order given by
  // user and remap operands coming from a previous stages.
  rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  }
  SmallVector<Value> predicates(maxStage + 1, nullptr);
  if (!peelEpilogue) {
    // Create a predicate for each stage except the last stage.
    for (unsigned i = 0; i < maxStage; i++) {
      // c = ub - (maxStage - i) * step
      predicates[i] = emitPredicateStageFn(rewriter, newForOp.getInductionVar(),
                                           ub, step, maxStage, i);
    }
  }
  for (Operation *op : opOrder) {
    int64_t useStage = stages[op];
    auto *newOp = rewriter.clone(*op, mapping);
    SmallVector<OpOperand *> operands;
    // Collect all the operands for the cloned op and its nested ops.
    op->walk([&operands](Operation *nestedOp) {
      for (OpOperand &operand : nestedOp->getOpOperands()) {
        operands.push_back(&operand);
      }
    });
    for (OpOperand *operand : operands) {
      Operation *nestedNewOp = mapping.lookup(operand->getOwner());
      // Special case for the induction variable uses. We replace it with a
      // version incremented based on the stage where it is used.
      if (operand->get() == forOp.getInductionVar()) {
        rewriter.setInsertionPoint(newOp);

        // offset = (maxStage - stages[op]) * step
        Type t = step.getType();
        Value offset = arith::MulIOp::create(
            rewriter, forOp.getLoc(), step,
            arith::ConstantOp::create(
                rewriter, forOp.getLoc(),
                rewriter.getIntegerAttr(t, maxStage - stages[op])));
        Value iv = arith::AddIOp::create(rewriter, forOp.getLoc(),
                                         newForOp.getInductionVar(), offset);
        nestedNewOp->setOperand(operand->getOperandNumber(), iv);
        rewriter.setInsertionPointAfter(newOp);
        continue;
      }
      Value source = operand->get();
      auto arg = dyn_cast<BlockArgument>(source);
      if (arg && arg.getOwner() == forOp.getBody()) {
        Value ret = forOp.getBody()->getTerminator()->getOperand(
            arg.getArgNumber() - 1);
        if (forOp.isDefinedOutsideOfLoop(ret)) {
          // Special case for values defined outside the loop accessed with
          // distance 1.
          if (useStage != maxStage) {
            nestedNewOp->setOperand(operand->getOperandNumber(), ret);
          }
          continue;
        }
        Operation *dep = ret.getDefiningOp();
        if (!dep)
          continue;
        auto stageDep = stages.find(dep);
        if (stageDep == stages.end() || stageDep->second == useStage)
          continue;
        // If the value is a loop carried value coming from stage N + 1 remap,
        // it will become a direct use.
        if (stageDep->second == useStage + 1) {
          nestedNewOp->setOperand(operand->getOperandNumber(),
                                  mapping.lookupOrDefault(ret));
          continue;
        }
        source = ret;
      }
      // For operands defined in a previous stage we need to remap it to use
      // the correct region argument. We look for the right version of the
      // Value based on the stage where it is used.
      Operation *def = source.getDefiningOp();
      if (!def)
        continue;
      auto stageDef = stages.find(def);
      if (stageDef == stages.end() || stageDef->second == useStage)
        continue;
      auto remap = loopArgMap.find(
          std::make_pair(operand->get(), useStage - stageDef->second));
      assert(remap != loopArgMap.end());
      nestedNewOp->setOperand(operand->getOperandNumber(),
                              newForOp.getRegionIterArgs()[remap->second]);
    }

    if (predicates[useStage]) {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      newOp = predicateFn(rewriter, newOp, predicates[useStage]);
      if (!newOp)
        return failure();
      // Remap the results to the new predicated one.
      for (auto values : llvm::zip(op->getResults(), newOp->getResults()))
        mapping.map(std::get<0>(values), std::get<1>(values));
    }
    if (annotateFn)
      annotateFn(newOp, triton::PipeliningOption::PipelinerPart::Kernel, 0);
  }

  // Collect the Values that need to be returned by the forOp. For each
  // value we need to have `LastUseStage - DefStage` number of versions
  // returned.
  // We create a mapping between original values and the associated loop
  // returned values that will be needed by the epilogue.
  llvm::SmallVector<Value> yieldOperands;
  for (OpOperand &yieldOperand :
       forOp.getBody()->getTerminator()->getOpOperands()) {
    Value source = mapping.lookupOrDefault(yieldOperand.get());
    // When we don't peel the epilogue and the yield value is used outside the
    // loop we need to make sure we return the version from numStages -
    // defStage.
    if (!peelEpilogue &&
        !forOp.getResult(yieldOperand.getOperandNumber()).use_empty()) {
      Operation *def = getDefiningOpAndDistance(yieldOperand.get()).first;
      if (def) {
        auto defStage = stages.find(def);
        if (defStage != stages.end() && defStage->second < maxStage) {
          Value pred = predicates[defStage->second];
          source = arith::SelectOp::create(
              rewriter, pred.getLoc(), pred, source,
              newForOp.getBody()
                  ->getArguments()[yieldOperand.getOperandNumber() + 1]);
        }
      }
    }
    yieldOperands.push_back(source);
  }

  for (auto &it : crossStageValues) {
    int64_t version = maxStage - it.second.lastUseStage + 1;
    unsigned numVersionReturned = it.second.lastUseStage - it.second.defStage;
    // add the original version to yield ops.
    // If there is a live range spanning across more than 2 stages we need to
    // add extra arg.
    for (unsigned i = 1; i < numVersionReturned; i++) {
      setValueMapping(it.first, newForOp->getResult(yieldOperands.size()),
                      version++);
      yieldOperands.push_back(
          newForOp.getBody()->getArguments()[yieldOperands.size() + 1 +
                                             newForOp.getNumInductionVars()]);
    }
    setValueMapping(it.first, newForOp->getResult(yieldOperands.size()),
                    version++);
    yieldOperands.push_back(mapping.lookupOrDefault(it.first));
  }
  // Map the yield operand to the forOp returned value.
  for (const auto &retVal :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
    Operation *def = retVal.value().getDefiningOp();
    auto defStage = stages.find(def);
    if (defStage == stages.end()) {
      for (unsigned int stage = 1; stage <= maxStage; stage++)
        setValueMapping(forOp.getRegionIterArgs()[retVal.index()],
                        retVal.value(), stage);
    } else if (defStage->second > 0) {
      setValueMapping(forOp.getRegionIterArgs()[retVal.index()],
                      newForOp->getResult(retVal.index()),
                      maxStage - defStage->second + 1);
    }
  }
  scf::YieldOp::create(rewriter, forOp.getLoc(), yieldOperands);
  return success();
}

LogicalResult
LoopPipelinerInternal::emitEpilogue(RewriterBase &rewriter,
                                    llvm::SmallVector<Value> &returnValues) {
  Location loc = forOp.getLoc();
  Type t = lb.getType();
  // Emit different versions of the induction variable. They will be
  // removed by dead code if not used.

  auto createConst = [&](int v) {
    return arith::ConstantOp::create(rewriter, loc,
                                     rewriter.getIntegerAttr(t, v));
  };

  // total_iterations = cdiv(range_diff, step);
  // - range_diff = ub - lb
  // - total_iterations = (range_diff + step + (step < 0 ? 1 : -1)) / step
  Value zero = createConst(0);
  Value one = createConst(1);
  Value stepLessZero = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::slt, step, zero);
  Value stepDecr = arith::SelectOp::create(rewriter, loc, stepLessZero, one,
                                           createConst(-1));

  Value rangeDiff = arith::SubIOp::create(rewriter, loc, ub, lb);
  Value rangeIncrStep = arith::AddIOp::create(rewriter, loc, rangeDiff, step);
  Value rangeDecr =
      arith::AddIOp::create(rewriter, loc, rangeIncrStep, stepDecr);
  Value totalIterations =
      arith::DivSIOp::create(rewriter, loc, rangeDecr, step);

  // If total_iters < max_stage, start the epilogue at zero to match the
  // ramp-up in the prologue.
  // start_iter = max(0, total_iters - max_stage)
  Value iterI = arith::SubIOp::create(rewriter, loc, totalIterations,
                                      createConst(maxStage));
  iterI = arith::MaxSIOp::create(rewriter, loc, zero, iterI);

  // Capture predicates for dynamic loops.
  SmallVector<Value> predicates(maxStage + 1);

  for (int64_t i = 1; i <= maxStage; i++) {
    // newLastIter = lb + step * iterI
    Value newlastIter = arith::AddIOp::create(
        rewriter, loc, lb, arith::MulIOp::create(rewriter, loc, step, iterI));

    setValueMapping(forOp.getInductionVar(), newlastIter, i);

    // increment to next iterI
    iterI = arith::AddIOp::create(rewriter, loc, iterI, one);

    if (dynamicLoop) {
      // Disable stages when `i` is greater than total_iters.
      // pred = total_iters >= i
      predicates[i] =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                totalIterations, createConst(i));
    }
  }

  // Emit `maxStage - 1` epilogue part that includes operations from stages
  // [i; maxStage].
  for (int64_t i = 1; i <= maxStage; i++) {
    SmallVector<std::pair<Value, unsigned>> returnMap(returnValues.size());
    for (Operation *op : opOrder) {
      if (stages[op] < i)
        continue;
      unsigned currentVersion = maxStage - stages[op] + i;
      unsigned nextVersion = currentVersion + 1;
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            auto it = valueMapping.find(newOperand->get());
            if (it != valueMapping.end()) {
              Value replacement = it->second[currentVersion];
              newOperand->set(replacement);
            }
          });
      if (dynamicLoop) {
        OpBuilder::InsertionGuard insertGuard(rewriter);
        newOp = predicateFn(rewriter, newOp, predicates[currentVersion]);
        if (!newOp)
          return failure();
      }
      if (annotateFn)
        annotateFn(newOp, triton::PipeliningOption::PipelinerPart::Epilogue,
                   i - 1);
      for (auto [opRes, newRes] :
           llvm::zip(op->getResults(), newOp->getResults())) {
        setValueMapping(opRes, newRes, currentVersion);
        // If the value is a loop carried dependency update the loop argument
        // mapping and keep track of the last version to replace the original
        // forOp uses.
        for (OpOperand &operand :
             forOp.getBody()->getTerminator()->getOpOperands()) {
          if (operand.get() != opRes)
            continue;
          // If the version is greater than maxStage it means it maps to the
          // original forOp returned value.
          unsigned ri = operand.getOperandNumber();
          returnValues[ri] = newRes;
          Value mapVal = forOp.getRegionIterArgs()[ri];
          returnMap[ri] = std::make_pair(mapVal, currentVersion);
          if (nextVersion <= maxStage)
            setValueMapping(mapVal, newRes, nextVersion);
        }
      }
    }
    if (dynamicLoop) {
      // Select return values from this stage (live outs) based on predication.
      // If the stage is valid select the peeled value, else use previous stage
      // value.
      for (auto pair : llvm::enumerate(returnValues)) {
        unsigned ri = pair.index();
        auto [mapVal, currentVersion] = returnMap[ri];
        if (mapVal) {
          unsigned nextVersion = currentVersion + 1;
          Value pred = predicates[currentVersion];
          Value prevValue = valueMapping[mapVal][currentVersion];
          auto selOp = arith::SelectOp::create(rewriter, loc, pred,
                                               pair.value(), prevValue);
          returnValues[ri] = selOp;
          if (nextVersion <= maxStage)
            setValueMapping(mapVal, selOp, nextVersion);
        }
      }
    }
  }
  return success();
}

void LoopPipelinerInternal::setValueMapping(Value key, Value el, int64_t idx) {
  auto it = valueMapping.find(key);
  // If the value is not in the map yet add a vector big enough to store all
  // versions.
  if (it == valueMapping.end())
    it =
        valueMapping
            .insert(std::make_pair(key, llvm::SmallVector<Value>(maxStage + 1)))
            .first;
  it->second[idx] = el;
}

} // namespace

FailureOr<ForOp>
mlir::triton::pipelineForLoop(RewriterBase &rewriter, ForOp forOp,
                              const triton::PipeliningOption &options,
                              bool *modifiedIR) {
  if (modifiedIR)
    *modifiedIR = false;
  LoopPipelinerInternal pipeliner;
  if (!pipeliner.initializeLoopInfo(forOp, options))
    return failure();

  if (modifiedIR)
    *modifiedIR = true;

  // 1. Emit prologue.
  if (failed(pipeliner.emitPrologue(rewriter)))
    return failure();

  // 2. Track values used across stages. When a value cross stages it will
  // need to be passed as loop iteration arguments.
  // We first collect the values that are used in a different stage than where
  // they are defined.
  llvm::MapVector<Value, LoopPipelinerInternal::LiverangeInfo>
      crossStageValues = pipeliner.analyzeCrossStageValues();

  // Mapping between original loop values used cross stage and the block
  // arguments associated after pipelining. A Value may map to several
  // arguments if its liverange spans across more than 2 stages.
  llvm::DenseMap<std::pair<Value, unsigned>, unsigned> loopArgMap;
  // 3. Create the new kernel loop and return the block arguments mapping.
  ForOp newForOp =
      pipeliner.createKernelLoop(crossStageValues, rewriter, loopArgMap);
  // Create the kernel block, order ops based on user choice and remap
  // operands.
  if (failed(pipeliner.createKernel(newForOp, crossStageValues, loopArgMap,
                                    rewriter)))
    return failure();

  llvm::SmallVector<Value> returnValues =
      newForOp.getResults().take_front(forOp->getNumResults());
  if (options.peelEpilogue) {
    // 4. Emit the epilogue after the new forOp.
    rewriter.setInsertionPointAfter(newForOp);
    if (failed(pipeliner.emitEpilogue(rewriter, returnValues)))
      return failure();
  }
  // 5. Erase the original loop and replace the uses with the epilogue output.
  if (forOp->getNumResults() > 0)
    rewriter.replaceOp(forOp, returnValues);
  else
    rewriter.eraseOp(forOp);

  return newForOp;
}

Value mlir::triton::emitPredicateForStage(RewriterBase &rewriter,
                                          Value inductionVar, Value upperBound,
                                          Value step, uint64_t maxStage,
                                          uint64_t stage) {
  auto loc = inductionVar.getLoc();
  auto type = inductionVar.getType();
  Value c = arith::SubIOp::create(
      rewriter, loc, upperBound,
      arith::MulIOp::create(
          rewriter, loc, step,
          arith::ConstantOp::create(
              rewriter, loc, rewriter.getIntegerAttr(type, maxStage - stage))));
  return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                               inductionVar, c);
}
