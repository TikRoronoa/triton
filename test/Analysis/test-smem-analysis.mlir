// RUN: triton-opt %s --smem-analysis 2>&1 | FileCheck %s
// CHECK: warp-level bank conflict: YES
// CHECK: current swizzle suboptimal
// CHECK: suggest: vec=8, perPhase=1, maxPhase=4
// CHECK: after fix: bank_conflict=YES

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#NO_SWIZZLE = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 16}>
#SWIZZLED = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @test(%A : !tt.ptr<f16>) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #NO_SWIZZLE, #ttg.shared_memory, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #SWIZZLED, #ttg.shared_memory, mutable>
    %a_load = ttg.local_load %a : !ttg.memdesc<64x32xf16, #NO_SWIZZLE, #ttg.shared_memory, mutable> -> tensor<64x32xf16, #blocked>
    %b_load = ttg.local_load %b : !ttg.memdesc<64x32xf16, #SWIZZLED, #ttg.shared_memory, mutable> -> tensor<64x32xf16, #blocked>
    ttg.local_dealloc %a : !ttg.memdesc<64x32xf16, #NO_SWIZZLE, #ttg.shared_memory, mutable>
    ttg.local_dealloc %b : !ttg.memdesc<64x32xf16, #SWIZZLED, #ttg.shared_memory, mutable>
    tt.return
  }
}
