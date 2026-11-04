#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared4 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared5 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @matmul_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    %12 = arith.cmpi slt, %arg0, %arg1 : index
    %13 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %14 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked1>
    %15 = ttg.async_copy_global_to_local %4, %13 mask %14 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable>
    %16 = ttg.async_commit_group tokens %15
    %17 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %18 = tt.splat %12 : i1 -> tensor<32x128xi1, #blocked>
    %19 = ttg.async_copy_global_to_local %9, %17 mask %18 other %cst_3 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
    %20 = ttg.async_commit_group tokens %19
    %21 = arith.addi %arg0, %arg2 : index
    %22 = arith.cmpi slt, %21, %arg1 : index
    %23 = tt.addptr %4, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %24 = tt.addptr %9, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %25 = ttg.memdesc_index %10[%c1_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %26 = tt.splat %22 : i1 -> tensor<128x32xi1, #blocked1>
    %27 = ttg.async_copy_global_to_local %23, %25 mask %26 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable>
    %28 = ttg.async_commit_group tokens %27
    %29 = ttg.memdesc_index %11[%c1_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %30 = tt.splat %22 : i1 -> tensor<32x128xi1, #blocked>
    %31 = ttg.async_copy_global_to_local %24, %29 mask %30 other %cst_3 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
    %32 = ttg.async_commit_group tokens %31
    %33:9 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %23, %arg7 = %24, %arg8 = %cst_2, %arg9 = %c1_i32, %arg10 = %c-1_i32, %arg11 = %16, %arg12 = %28, %arg13 = %20, %arg14 = %32) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      %35 = arith.muli %arg2, %c2 : index
      %36 = arith.subi %arg1, %35 : index
      %37 = arith.cmpi slt, %arg5, %36 : index
      %38 = arith.addi %arg10, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.select %39, %c0_i32, %38 : i32
      %41 = ttg.async_wait %arg11, %arg13 {num = 2 : i32}
      %42 = ttg.memdesc_index %10[%40] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %43 = ttg.local_load %42 token %41 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %44 = ttg.memdesc_index %11[%40] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      %45 = ttg.local_load %44 token %41 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %46 = arith.mulf %45, %cst : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %47 = tt.dot %43, %46, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %48 = tt.addptr %arg6, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %49 = tt.addptr %arg7, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %50 = arith.addi %arg9, %c1_i32 : i32
      %51 = arith.cmpi sge, %50, %c2_i32 : i32
      %52 = arith.select %51, %c0_i32, %50 : i32
      %53 = ttg.memdesc_index %10[%52] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %54 = tt.splat %37 : i1 -> tensor<128x32xi1, #blocked1>
      %55 = ttg.async_copy_global_to_local %48, %53 mask %54 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable>
      %56 = ttg.async_commit_group tokens %55
      %57 = ttg.memdesc_index %11[%52] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      %58 = tt.splat %37 : i1 -> tensor<32x128xi1, #blocked>
      %59 = ttg.async_copy_global_to_local %49, %57 mask %58 other %cst_3 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
      %60 = ttg.async_commit_group tokens %59
      scf.yield %48, %49, %47, %52, %40, %arg12, %56, %arg14, %60 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    %34 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %11 : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    ttg.local_dealloc %10 : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    tt.return %33#2 : tensor<128x128xf32, #mma>
  }
  tt.func @matmul_loop_nested(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %cst_3) -> (tensor<128x128xf32, #mma>) {
      %1 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
      %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
      %4 = tt.broadcast %3 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
      %5 = tt.addptr %1, %4 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %6 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
      %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %9 = tt.broadcast %8 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %10 = tt.addptr %6, %9 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %11 = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
      %12 = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
      %13 = arith.cmpi slt, %arg0, %arg1 : index
      %14 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %15 = tt.splat %13 : i1 -> tensor<128x32xi1, #blocked1>
      %16 = ttg.async_copy_global_to_local %5, %14 mask %15 other %cst_2 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable>
      %17 = ttg.async_commit_group tokens %16
      %18 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      %19 = tt.splat %13 : i1 -> tensor<32x128xi1, #blocked>
      %20 = ttg.async_copy_global_to_local %10, %18 mask %19 other %cst_1 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
      %21 = ttg.async_commit_group tokens %20
      %22 = arith.addi %arg0, %arg2 : index
      %23 = arith.cmpi slt, %22, %arg1 : index
      %24 = tt.addptr %5, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %25 = tt.addptr %10, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %26 = ttg.memdesc_index %11[%c1_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %27 = tt.splat %23 : i1 -> tensor<128x32xi1, #blocked1>
      %28 = ttg.async_copy_global_to_local %24, %26 mask %27 other %cst_2 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable>
      %29 = ttg.async_commit_group tokens %28
      %30 = ttg.memdesc_index %12[%c1_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      %31 = tt.splat %23 : i1 -> tensor<32x128xi1, #blocked>
      %32 = ttg.async_copy_global_to_local %25, %30 mask %31 other %cst_1 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
      %33 = ttg.async_commit_group tokens %32
      %34:9 = scf.for %arg7 = %arg0 to %arg1 step %arg2 iter_args(%arg8 = %24, %arg9 = %25, %arg10 = %arg6, %arg11 = %c1_i32, %arg12 = %c-1_i32, %arg13 = %17, %arg14 = %29, %arg15 = %21, %arg16 = %33) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
        %36 = arith.muli %arg2, %c2 : index
        %37 = arith.subi %arg1, %36 : index
        %38 = arith.cmpi slt, %arg7, %37 : index
        %39 = arith.addi %arg12, %c1_i32 : i32
        %40 = arith.cmpi sge, %39, %c2_i32 : i32
        %41 = arith.select %40, %c0_i32, %39 : i32
        %42 = ttg.async_wait %arg13, %arg15 {num = 2 : i32}
        %43 = ttg.memdesc_index %11[%41] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
        %44 = ttg.local_load %43 token %42 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
        %45 = ttg.memdesc_index %12[%41] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
        %46 = ttg.local_load %45 token %42 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
        %47 = tt.dot %44, %46, %arg10 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
        %48 = tt.addptr %arg8, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
        %49 = tt.addptr %arg9, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
        %50 = arith.addi %arg11, %c1_i32 : i32
        %51 = arith.cmpi sge, %50, %c2_i32 : i32
        %52 = arith.select %51, %c0_i32, %50 : i32
        %53 = ttg.memdesc_index %11[%52] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
        %54 = tt.splat %38 : i1 -> tensor<128x32xi1, #blocked1>
        %55 = ttg.async_copy_global_to_local %48, %53 mask %54 other %cst_2 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable>
        %56 = ttg.async_commit_group tokens %55
        %57 = ttg.memdesc_index %12[%52] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
        %58 = tt.splat %38 : i1 -> tensor<32x128xi1, #blocked>
        %59 = ttg.async_copy_global_to_local %49, %57 mask %58 other %cst_1 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
        %60 = ttg.async_commit_group tokens %59
        scf.yield %48, %49, %47, %52, %41, %arg14, %56, %arg16, %60 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      }
      %35 = ttg.async_wait {num = 0 : i32}
      ttg.local_dealloc %12 : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
      ttg.local_dealloc %11 : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
      scf.yield %34#2 : tensor<128x128xf32, #mma>
    }
    tt.return %0 : tensor<128x128xf32, #mma>
  }
  tt.func @matmul_loop_single_pipeline(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10 = tt.load %4 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %11 = ttg.convert_layout %10 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    %13 = arith.cmpi slt, %arg0, %arg1 : index
    %14 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %15 = tt.splat %13 : i1 -> tensor<32x128xi1, #blocked>
    %16 = ttg.async_copy_global_to_local %9, %14 mask %15 other %cst_1 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
    %17 = ttg.async_commit_group tokens %16
    %18 = arith.addi %arg0, %arg2 : index
    %19 = arith.cmpi slt, %18, %arg1 : index
    %20 = tt.addptr %9, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %21 = ttg.memdesc_index %12[%c1_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %22 = tt.splat %19 : i1 -> tensor<32x128xi1, #blocked>
    %23 = ttg.async_copy_global_to_local %20, %21 mask %22 other %cst_1 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
    %24 = ttg.async_commit_group tokens %23
    %25:6 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %20, %arg7 = %cst_0, %arg8 = %c1_i32, %arg9 = %c-1_i32, %arg10 = %17, %arg11 = %24) -> (tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token) {
      %27 = arith.muli %arg2, %c2 : index
      %28 = arith.subi %arg1, %27 : index
      %29 = arith.cmpi slt, %arg5, %28 : index
      %30 = arith.addi %arg9, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c2_i32 : i32
      %32 = arith.select %31, %c0_i32, %30 : i32
      %33 = ttg.async_wait %arg10 {num = 1 : i32}
      %34 = ttg.memdesc_index %12[%32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      %35 = ttg.local_load %34 token %33 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %36 = tt.dot %11, %35, %arg7 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %37 = tt.addptr %arg6, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %38 = arith.addi %arg8, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.select %39, %c0_i32, %38 : i32
      %41 = ttg.memdesc_index %12[%40] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      %42 = tt.splat %29 : i1 -> tensor<32x128xi1, #blocked>
      %43 = ttg.async_copy_global_to_local %37, %41 mask %42 other %cst_1 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
      %44 = ttg.async_commit_group tokens %43
      scf.yield %37, %36, %40, %32, %arg11, %44 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token
    }
    %26 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %12 : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    tt.return %25#1 : tensor<128x128xf32, #mma>
  }
  tt.func @indirect_bmm_scalar(%arg0: i64 {tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg3: !tt.ptr<i64>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    %3 = arith.cmpi sgt, %arg1, %c0 : index
    %4 = tt.splat %arg3 : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>, #blocked2>
    %5 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %6 = tt.splat %3 : i1 -> tensor<1xi1, #blocked2>
    %7 = ttg.async_copy_global_to_local %4, %5 mask %6 : tensor<1x!tt.ptr<i64>, #blocked2> -> <1xi64, #shared3, #smem, mutable>
    %8 = ttg.async_commit_group tokens %7
    %9 = arith.cmpi sgt, %arg1, %c1 : index
    %10 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
    %11 = tt.splat %3 : i1 -> tensor<16x16xi1, #blocked1>
    %12 = ttg.async_copy_global_to_local %arg2, %10 mask %11 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared2, #smem, mutable>
    %13 = ttg.async_commit_group tokens %12
    %14 = ttg.async_wait %8 {num = 1 : i32}
    %15 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %16 = ttg.local_load %15 token %14 : !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> tensor<1xi64, #blocked2>
    %17 = tt.unsplat %16 : tensor<1xi64, #blocked2>
    %18 = arith.muli %arg0, %17 : i64
    %19 = tt.splat %18 : i64 -> tensor<16x16xi64, #blocked>
    %20 = tt.addptr %arg5, %19 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %21 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
    %22 = tt.splat %3 : i1 -> tensor<16x16xi1, #blocked>
    %23 = ttg.async_copy_global_to_local %20, %21 mask %22 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared2, #smem, mutable>
    %24 = ttg.async_commit_group tokens %23
    %25 = tt.addptr %arg3, %c1_i32 : !tt.ptr<i64>, i32
    %26 = tt.splat %25 : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>, #blocked2>
    %27 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
    %28 = tt.splat %9 : i1 -> tensor<1xi1, #blocked2>
    %29 = ttg.async_copy_global_to_local %26, %27 mask %28 : tensor<1x!tt.ptr<i64>, #blocked2> -> <1xi64, #shared3, #smem, mutable>
    %30 = ttg.async_commit_group tokens %29
    %31:8 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %25, %arg10 = %c0_i32, %arg11 = %c0_i32, %arg12 = %13, %arg13 = %24, %arg14 = %30) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, !tt.ptr<i64>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      %33 = arith.subi %arg1, %c2 : index
      %34 = arith.cmpi slt, %arg6, %33 : index
      %35 = arith.subi %arg1, %c1 : index
      %36 = arith.cmpi slt, %arg6, %35 : index
      %37 = ttg.async_wait %arg12, %arg13 {num = 1 : i32}
      %38 = ttg.memdesc_index %0[%arg11] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %39 = ttg.local_load %38 token %37 : !ttg.memdesc<16x16xf16, #shared2, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %40 = ttg.memdesc_index %2[%arg11] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %41 = ttg.local_load %40 token %37 : !ttg.memdesc<16x16xf16, #shared2, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %42 = tt.dot %39, %41, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %43 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %44 = arith.addi %arg11, %c1_i32 : i32
      %45 = arith.cmpi sge, %44, %c1_i32 : i32
      %46 = arith.select %45, %c0_i32, %44 : i32
      %47 = ttg.memdesc_index %0[%arg10] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %48 = tt.splat %36 : i1 -> tensor<16x16xi1, #blocked1>
      %49 = ttg.async_copy_global_to_local %43, %47 mask %48 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared2, #smem, mutable>
      %50 = ttg.async_commit_group tokens %49
      %51 = ttg.async_wait %arg14 {num = 1 : i32}
      %52 = ttg.memdesc_index %1[%46] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %53 = ttg.local_load %52 token %51 : !ttg.memdesc<1xi64, #shared3, #smem, mutable> -> tensor<1xi64, #blocked2>
      %54 = tt.unsplat %53 : tensor<1xi64, #blocked2>
      %55 = arith.muli %arg0, %54 : i64
      %56 = tt.splat %55 : i64 -> tensor<16x16xi64, #blocked>
      %57 = tt.addptr %arg5, %56 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %58 = ttg.memdesc_index %2[%arg10] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %59 = tt.splat %36 : i1 -> tensor<16x16xi1, #blocked>
      %60 = ttg.async_copy_global_to_local %57, %58 mask %59 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared2, #smem, mutable>
      %61 = ttg.async_commit_group tokens %60
      %62 = tt.addptr %arg9, %c1_i32 : !tt.ptr<i64>, i32
      %63 = arith.addi %arg10, %c1_i32 : i32
      %64 = arith.cmpi sge, %63, %c1_i32 : i32
      %65 = arith.select %64, %c0_i32, %63 : i32
      %66 = tt.splat %62 : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>, #blocked2>
      %67 = ttg.memdesc_index %1[%65] : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable> -> !ttg.memdesc<1xi64, #shared3, #smem, mutable>
      %68 = tt.splat %34 : i1 -> tensor<1xi1, #blocked2>
      %69 = ttg.async_copy_global_to_local %66, %67 mask %68 : tensor<1x!tt.ptr<i64>, #blocked2> -> <1xi64, #shared3, #smem, mutable>
      %70 = ttg.async_commit_group tokens %69
      scf.yield %42, %43, %62, %65, %46, %50, %61, %70 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, !tt.ptr<i64>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.num_stages = 3 : i32}
    %32 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %2 : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<1x1xi64, #shared3, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    tt.return %31#0 : tensor<16x16xf32, #mma>
  }
  tt.func @indirect_bmm_scalar_dist_one(%arg0: i64 {tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg3: !tt.ptr<i64>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.load %arg3 : !tt.ptr<i64>
    %1 = tt.addptr %arg3, %c1_i32 : !tt.ptr<i64>, i32
    %2 = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable>
    %4 = arith.cmpi sgt, %arg1, %c0 : index
    %5 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
    %6 = tt.splat %4 : i1 -> tensor<16x16xi1, #blocked1>
    %7 = ttg.async_copy_global_to_local %arg2, %5 mask %6 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared2, #smem, mutable>
    %8 = ttg.async_commit_group tokens %7
    %9 = tt.load %1, %4 : !tt.ptr<i64>
    %10 = arith.muli %arg0, %0 : i64
    %11 = tt.splat %10 : i64 -> tensor<16x16xi64, #blocked>
    %12 = tt.addptr %arg5, %11 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %13 = ttg.memdesc_index %3[%c0_i32] : !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
    %14 = tt.splat %4 : i1 -> tensor<16x16xi1, #blocked>
    %15 = ttg.async_copy_global_to_local %12, %13 mask %14 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared2, #smem, mutable>
    %16 = ttg.async_commit_group tokens %15
    %17 = tt.addptr %1, %c1_i32 : !tt.ptr<i64>, i32
    %18 = arith.cmpi sgt, %arg1, %c1 : index
    %19 = tt.addptr %arg2, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
    %20 = ttg.memdesc_index %2[%c1_i32] : !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
    %21 = tt.splat %18 : i1 -> tensor<16x16xi1, #blocked1>
    %22 = ttg.async_copy_global_to_local %19, %20 mask %21 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared2, #smem, mutable>
    %23 = ttg.async_commit_group tokens %22
    %24 = tt.load %17, %18 : !tt.ptr<i64>
    %25 = arith.muli %arg0, %9 : i64
    %26 = tt.splat %25 : i64 -> tensor<16x16xi64, #blocked>
    %27 = tt.addptr %arg5, %26 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %28 = ttg.memdesc_index %3[%c1_i32] : !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
    %29 = tt.splat %18 : i1 -> tensor<16x16xi1, #blocked>
    %30 = ttg.async_copy_global_to_local %27, %28 mask %29 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared2, #smem, mutable>
    %31 = ttg.async_commit_group tokens %30
    %32 = tt.addptr %17, %c1_i32 : !tt.ptr<i64>, i32
    %33:10 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %19, %arg9 = %32, %arg10 = %24, %arg11 = %c1_i32, %arg12 = %c-1_i32, %arg13 = %8, %arg14 = %23, %arg15 = %16, %arg16 = %31) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, !tt.ptr<i64>, i64, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      %35 = arith.subi %arg1, %c2 : index
      %36 = arith.cmpi slt, %arg6, %35 : index
      %37 = arith.addi %arg12, %c1_i32 : i32
      %38 = arith.cmpi sge, %37, %c2_i32 : i32
      %39 = arith.select %38, %c0_i32, %37 : i32
      %40 = ttg.async_wait %arg13, %arg15 {num = 2 : i32}
      %41 = ttg.memdesc_index %2[%39] : !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %42 = ttg.local_load %41 token %40 : !ttg.memdesc<16x16xf16, #shared2, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %43 = ttg.memdesc_index %3[%39] : !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %44 = ttg.local_load %43 token %40 : !ttg.memdesc<16x16xf16, #shared2, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %45 = tt.dot %42, %44, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %46 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %47 = arith.addi %arg11, %c1_i32 : i32
      %48 = arith.cmpi sge, %47, %c2_i32 : i32
      %49 = arith.select %48, %c0_i32, %47 : i32
      %50 = ttg.memdesc_index %2[%49] : !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %51 = tt.splat %36 : i1 -> tensor<16x16xi1, #blocked1>
      %52 = ttg.async_copy_global_to_local %46, %50 mask %51 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared2, #smem, mutable>
      %53 = ttg.async_commit_group tokens %52
      %54 = tt.load %arg9, %36 : !tt.ptr<i64>
      %55 = arith.muli %arg0, %arg10 : i64
      %56 = tt.splat %55 : i64 -> tensor<16x16xi64, #blocked>
      %57 = tt.addptr %arg5, %56 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %58 = ttg.memdesc_index %3[%49] : !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %59 = tt.splat %36 : i1 -> tensor<16x16xi1, #blocked>
      %60 = ttg.async_copy_global_to_local %57, %58 mask %59 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared2, #smem, mutable>
      %61 = ttg.async_commit_group tokens %60
      %62 = tt.addptr %arg9, %c1_i32 : !tt.ptr<i64>, i32
      scf.yield %45, %46, %62, %54, %49, %39, %arg14, %53, %arg16, %61 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, !tt.ptr<i64>, i64, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    %34 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %3 : !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable>
    ttg.local_dealloc %2 : !ttg.memdesc<2x16x16xf16, #shared2, #smem, mutable>
    tt.return %33#0 : tensor<16x16xf32, #mma>
  }
  tt.func @indirect_bmm_vector(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg3: tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<1> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x16xi64, #shared3, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    %3 = arith.cmpi sgt, %arg1, %c0 : index
    %4 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x16xi64, #shared3, #smem, mutable> -> !ttg.memdesc<16xi64, #shared3, #smem, mutable>
    %5 = tt.splat %3 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = ttg.async_copy_global_to_local %arg3, %4 mask %5 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>> -> <16xi64, #shared3, #smem, mutable>
    %7 = ttg.async_commit_group tokens %6
    %8 = arith.cmpi sgt, %arg1, %c1 : index
    %9 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
    %10 = tt.splat %3 : i1 -> tensor<16x16xi1, #blocked1>
    %11 = ttg.async_copy_global_to_local %arg2, %9 mask %10 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared2, #smem, mutable>
    %12 = ttg.async_commit_group tokens %11
    %13 = ttg.async_wait %7 {num = 1 : i32}
    %14 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x16xi64, #shared3, #smem, mutable> -> !ttg.memdesc<16xi64, #shared3, #smem, mutable>
    %15 = ttg.local_load %14 token %13 : !ttg.memdesc<16xi64, #shared3, #smem, mutable> -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
    %17 = tt.broadcast %16 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
    %18 = arith.muli %arg0, %17 : tensor<16x16xi64, #blocked>
    %19 = tt.addptr %arg5, %18 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %20 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
    %21 = tt.splat %3 : i1 -> tensor<16x16xi1, #blocked>
    %22 = ttg.async_copy_global_to_local %19, %20 mask %21 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared2, #smem, mutable>
    %23 = ttg.async_commit_group tokens %22
    %24 = tt.addptr %arg3, %cst_0 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %25 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x16xi64, #shared3, #smem, mutable> -> !ttg.memdesc<16xi64, #shared3, #smem, mutable>
    %26 = tt.splat %8 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
    %27 = ttg.async_copy_global_to_local %24, %25 mask %26 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>> -> <16xi64, #shared3, #smem, mutable>
    %28 = ttg.async_commit_group tokens %27
    %29:8 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %24, %arg10 = %c0_i32, %arg11 = %c0_i32, %arg12 = %12, %arg13 = %23, %arg14 = %28) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      %31 = arith.subi %arg1, %c2 : index
      %32 = arith.cmpi slt, %arg6, %31 : index
      %33 = arith.subi %arg1, %c1 : index
      %34 = arith.cmpi slt, %arg6, %33 : index
      %35 = ttg.async_wait %arg12, %arg13 {num = 1 : i32}
      %36 = ttg.memdesc_index %0[%arg11] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %37 = ttg.local_load %36 token %35 : !ttg.memdesc<16x16xf16, #shared2, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %38 = ttg.memdesc_index %2[%arg11] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %39 = ttg.local_load %38 token %35 : !ttg.memdesc<16x16xf16, #shared2, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %40 = tt.dot %37, %39, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %41 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %42 = arith.addi %arg11, %c1_i32 : i32
      %43 = arith.cmpi sge, %42, %c1_i32 : i32
      %44 = arith.select %43, %c0_i32, %42 : i32
      %45 = ttg.memdesc_index %0[%arg10] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %46 = tt.splat %34 : i1 -> tensor<16x16xi1, #blocked1>
      %47 = ttg.async_copy_global_to_local %41, %45 mask %46 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared2, #smem, mutable>
      %48 = ttg.async_commit_group tokens %47
      %49 = ttg.async_wait %arg14 {num = 1 : i32}
      %50 = ttg.memdesc_index %1[%44] : !ttg.memdesc<1x16xi64, #shared3, #smem, mutable> -> !ttg.memdesc<16xi64, #shared3, #smem, mutable>
      %51 = ttg.local_load %50 token %49 : !ttg.memdesc<16xi64, #shared3, #smem, mutable> -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %52 = tt.expand_dims %51 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %53 = tt.broadcast %52 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
      %54 = arith.muli %arg0, %53 : tensor<16x16xi64, #blocked>
      %55 = tt.addptr %arg5, %54 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %56 = ttg.memdesc_index %2[%arg10] : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared2, #smem, mutable>
      %57 = tt.splat %34 : i1 -> tensor<16x16xi1, #blocked>
      %58 = ttg.async_copy_global_to_local %55, %56 mask %57 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared2, #smem, mutable>
      %59 = ttg.async_commit_group tokens %58
      %60 = tt.addptr %arg9, %cst_0 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %61 = arith.addi %arg10, %c1_i32 : i32
      %62 = arith.cmpi sge, %61, %c1_i32 : i32
      %63 = arith.select %62, %c0_i32, %61 : i32
      %64 = ttg.memdesc_index %1[%63] : !ttg.memdesc<1x16xi64, #shared3, #smem, mutable> -> !ttg.memdesc<16xi64, #shared3, #smem, mutable>
      %65 = tt.splat %32 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
      %66 = ttg.async_copy_global_to_local %60, %64 mask %65 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>> -> <16xi64, #shared3, #smem, mutable>
      %67 = ttg.async_commit_group tokens %66
      scf.yield %40, %41, %60, %63, %44, %48, %59, %67 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.num_stages = 3 : i32}
    %30 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %2 : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<1x16xi64, #shared3, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<1x16x16xf16, #shared2, #smem, mutable>
    tt.return %29#0 : tensor<16x16xf32, #mma>
  }
  tt.func @post_load_inv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c898 = arith.constant 898 : index
    %cst = arith.constant dense<32> : tensor<32x32xi32, #blocked1>
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c900 = arith.constant 900 : index
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked1>
    %0 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %3 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked1>
    %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %5 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<2x32x32xf32, #shared4, #smem, mutable>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<2x32x32xf32, #shared5, #smem, mutable>
    %8 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked1>
    %9 = arith.cmpi slt, %0, %8 : tensor<1x32xi32, #blocked1>
    %10 = tt.broadcast %9 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %11 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<2x32x32xf32, #shared4, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared4, #smem, mutable>
    %12 = ttg.async_copy_global_to_local %1, %11 mask %10 other %cst_1 : tensor<32x32x!tt.ptr<f32>, #blocked1> -> <32x32xf32, #shared4, #smem, mutable>
    %13 = ttg.async_commit_group tokens %12
    %14 = tt.splat %arg5 : i32 -> tensor<32x1xi32, #blocked1>
    %15 = arith.cmpi slt, %3, %14 : tensor<32x1xi32, #blocked1>
    %16 = tt.broadcast %15 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %17 = ttg.memdesc_index %7[%c0_i32] : !ttg.memdesc<2x32x32xf32, #shared5, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared5, #smem, mutable>
    %18 = ttg.async_copy_global_to_local %2, %17 mask %16 other %cst_1 : tensor<32x32x!tt.ptr<f32>, #blocked1> -> <32x32xf32, #shared5, #smem, mutable>
    %19 = ttg.async_commit_group tokens %18
    %20 = tt.addptr %4, %cst : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
    %21 = arith.muli %arg7, %c32_i32 : i32
    %22 = tt.splat %21 : i32 -> tensor<32x32xi32, #blocked1>
    %23 = tt.addptr %5, %22 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
    %24 = arith.subi %arg5, %c32_i32 : i32
    %25 = tt.splat %24 : i32 -> tensor<1x32xi32, #blocked1>
    %26 = arith.cmpi slt, %0, %25 : tensor<1x32xi32, #blocked1>
    %27 = tt.broadcast %26 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %28 = ttg.memdesc_index %6[%c1_i32] : !ttg.memdesc<2x32x32xf32, #shared4, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared4, #smem, mutable>
    %29 = ttg.async_copy_global_to_local %20, %28 mask %27 other %cst_1 : tensor<32x32x!tt.ptr<f32>, #blocked1> -> <32x32xf32, #shared4, #smem, mutable>
    %30 = ttg.async_commit_group tokens %29
    %31 = tt.splat %24 : i32 -> tensor<32x1xi32, #blocked1>
    %32 = arith.cmpi slt, %3, %31 : tensor<32x1xi32, #blocked1>
    %33 = tt.broadcast %32 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %34 = ttg.memdesc_index %7[%c1_i32] : !ttg.memdesc<2x32x32xf32, #shared5, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared5, #smem, mutable>
    %35 = ttg.async_copy_global_to_local %23, %34 mask %33 other %cst_1 : tensor<32x32x!tt.ptr<f32>, #blocked1> -> <32x32xf32, #shared5, #smem, mutable>
    %36 = ttg.async_commit_group tokens %35
    %37:7 = scf.for %arg9 = %c0 to %c900 step %c1 iter_args(%arg10 = %cst_0, %arg11 = %c1_i32, %arg12 = %c-1_i32, %arg13 = %13, %arg14 = %30, %arg15 = %19, %arg16 = %36) -> (tensor<32x32xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      %39 = arith.cmpi slt, %arg9, %c898 : index
      %40 = arith.addi %arg12, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.select %41, %c0_i32, %40 : i32
      %43 = ttg.async_wait %arg13, %arg15 {num = 2 : i32}
      %44 = ttg.memdesc_index %6[%42] : !ttg.memdesc<2x32x32xf32, #shared4, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared4, #smem, mutable>
      %45 = ttg.local_load %44 token %43 : !ttg.memdesc<32x32xf32, #shared4, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %46 = ttg.memdesc_index %7[%42] : !ttg.memdesc<2x32x32xf32, #shared5, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared5, #smem, mutable>
      %47 = ttg.local_load %46 token %43 : !ttg.memdesc<32x32xf32, #shared5, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %48 = tt.dot %45, %47, %arg10, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %49 = arith.addi %arg9, %c1 : index
      %50 = arith.index_cast %49 : index to i32
      %51 = arith.addi %50, %c1_i32 : i32
      %52 = arith.muli %51, %c32_i32 : i32
      %53 = tt.splat %52 : i32 -> tensor<32x32xi32, #blocked1>
      %54 = tt.addptr %4, %53 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
      %55 = arith.muli %52, %arg7 : i32
      %56 = tt.splat %55 : i32 -> tensor<32x32xi32, #blocked1>
      %57 = tt.addptr %5, %56 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
      %58 = arith.addi %arg11, %c1_i32 : i32
      %59 = arith.cmpi sge, %58, %c2_i32 : i32
      %60 = arith.select %59, %c0_i32, %58 : i32
      %61 = arith.addi %arg9, %c2 : index
      %62 = arith.index_cast %61 : index to i32
      %63 = arith.muli %62, %c32_i32 : i32
      %64 = arith.subi %arg5, %63 : i32
      %65 = tt.splat %64 : i32 -> tensor<1x32xi32, #blocked1>
      %66 = arith.cmpi slt, %0, %65 : tensor<1x32xi32, #blocked1>
      %67 = tt.broadcast %66 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %68 = ttg.memdesc_index %6[%60] : !ttg.memdesc<2x32x32xf32, #shared4, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared4, #smem, mutable>
      %69 = tt.splat %39 : i1 -> tensor<32x32xi1, #blocked1>
      %70 = arith.andi %69, %67 : tensor<32x32xi1, #blocked1>
      %71 = ttg.async_copy_global_to_local %54, %68 mask %70 other %cst_1 : tensor<32x32x!tt.ptr<f32>, #blocked1> -> <32x32xf32, #shared4, #smem, mutable>
      %72 = ttg.async_commit_group tokens %71
      %73 = tt.splat %64 : i32 -> tensor<32x1xi32, #blocked1>
      %74 = arith.cmpi slt, %3, %73 : tensor<32x1xi32, #blocked1>
      %75 = tt.broadcast %74 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %76 = ttg.memdesc_index %7[%60] : !ttg.memdesc<2x32x32xf32, #shared5, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared5, #smem, mutable>
      %77 = tt.splat %39 : i1 -> tensor<32x32xi1, #blocked1>
      %78 = arith.andi %77, %75 : tensor<32x32xi1, #blocked1>
      %79 = ttg.async_copy_global_to_local %57, %76 mask %78 other %cst_1 : tensor<32x32x!tt.ptr<f32>, #blocked1> -> <32x32xf32, #shared5, #smem, mutable>
      %80 = ttg.async_commit_group tokens %79
      scf.yield %48, %60, %42, %arg14, %72, %arg16, %80 : tensor<32x32xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    %38 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %7 : !ttg.memdesc<2x32x32xf32, #shared5, #smem, mutable>
    ttg.local_dealloc %6 : !ttg.memdesc<2x32x32xf32, #shared4, #smem, mutable>
    tt.return %37#0 : tensor<32x32xf32, #mma>
  }
  tt.func @cross_iter_dep(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked1>
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %5 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #blocked1>
    %6 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked1>
    %7 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %8:5 = scf.for %arg9 = %c0 to %c32 step %c1 iter_args(%arg10 = %cst, %arg11 = %0, %arg12 = %1, %arg13 = %3, %arg14 = %4) -> (tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>) {
      %9 = arith.index_cast %arg9 : index to i32
      %10 = arith.muli %9, %c32_i32 : i32
      %11 = arith.subi %arg5, %10 : i32
      %12 = tt.splat %11 : i32 -> tensor<1x32xi32, #blocked1>
      %13 = arith.cmpi slt, %5, %12 : tensor<1x32xi32, #blocked1>
      %14 = tt.broadcast %13 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %15 = tt.load %arg11, %14, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %16 = tt.splat %11 : i32 -> tensor<32x1xi32, #blocked1>
      %17 = arith.cmpi slt, %6, %16 : tensor<32x1xi32, #blocked1>
      %18 = tt.broadcast %17 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %19 = tt.load %arg12, %18, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %20 = ttg.convert_layout %15 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %21 = ttg.convert_layout %19 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %22 = tt.dot %20, %21, %arg10, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %23 = arith.index_cast %arg9 : index to i32
      %24 = arith.addi %23, %c2_i32 : i32
      %25 = arith.muli %24, %c32_i32 : i32
      %26 = tt.splat %25 : i32 -> tensor<32x32xi32, #blocked1>
      %27 = tt.addptr %7, %26 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
      %28 = arith.muli %25, %arg7 : i32
      %29 = tt.splat %28 : i32 -> tensor<32x32xi32, #blocked1>
      %30 = tt.addptr %2, %29 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
      scf.yield %22, %arg13, %arg14, %27, %30 : tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>
    }
    tt.return %8#0 : tensor<32x32xf32, #mma>
  }
  tt.func @dep_arg_two_uses(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<64> : tensor<32x128xi64, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant dense<64> : tensor<1x32xi64, #blocked1>
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c32 = arith.constant 32 : index
    %c100 = arith.constant 100 : index
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %5 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.addptr %arg1, %c32_i32 : !tt.ptr<i32>, i32
    %7:5 = scf.for %arg3 = %c0 to %c100 step %c32 iter_args(%arg4 = %4, %arg5 = %3, %arg6 = %6, %arg7 = %cst_2, %arg8 = %5) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, !tt.ptr<i32>, tensor<128x128xf32, #mma>, tensor<32x128x!tt.ptr<f16>, #blocked>) {
      %8 = arith.subi %c100, %arg3 : index
      %9 = arith.index_cast %8 : index to i32
      %10 = tt.splat %9 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %11 = tt.splat %9 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = arith.cmpi slt, %1, %10 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %13 = arith.cmpi slt, %2, %11 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = tt.expand_dims %12 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi1, #blocked1>
      %15 = tt.expand_dims %13 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi1, #blocked>
      %16 = tt.expand_dims %arg5 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
      %17 = tt.expand_dims %arg5 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
      %18 = arith.extsi %16 : tensor<1x32xi32, #blocked1> to tensor<1x32xi64, #blocked1>
      %19 = arith.extsi %17 : tensor<1x32xi32, #blocked1> to tensor<1x32xi64, #blocked1>
      %20 = arith.muli %18, %cst_1 : tensor<1x32xi64, #blocked1>
      %21 = arith.muli %19, %cst_1 : tensor<1x32xi64, #blocked1>
      %22 = tt.broadcast %20 : tensor<1x32xi64, #blocked1> -> tensor<128x32xi64, #blocked1>
      %23 = tt.broadcast %21 : tensor<1x32xi64, #blocked1> -> tensor<128x32xi64, #blocked1>
      %24 = tt.addptr %arg4, %22 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi64, #blocked1>
      %25 = tt.addptr %arg4, %23 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi64, #blocked1>
      %26 = tt.broadcast %14 : tensor<1x32xi1, #blocked1> -> tensor<128x32xi1, #blocked1>
      %27 = tt.load %25, %26 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %28 = tt.splat %arg6 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %29 = tt.addptr %28, %0 : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %30 = tt.load %29 : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %31 = tt.addptr %arg6, %c32_i32 : !tt.ptr<i32>, i32
      %32 = tt.broadcast %15 : tensor<32x1xi1, #blocked> -> tensor<32x128xi1, #blocked>
      %33 = tt.load %arg8, %32, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>
      %34 = ttg.convert_layout %27 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %35 = ttg.convert_layout %33 : tensor<32x128xf16, #blocked> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %36 = tt.dot %34, %35, %arg7 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %37 = tt.addptr %arg8, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi64, #blocked>
      scf.yield %24, %30, %31, %36, %37 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, !tt.ptr<i32>, tensor<128x128xf32, #mma>, tensor<32x128x!tt.ptr<f16>, #blocked>
    }
    tt.return %7#3 : tensor<128x128xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @load_two_users_incompatible_layouts(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %3 = tt.broadcast %1 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %4 = tt.addptr %2, %3 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %5 = tt.load %4 : tensor<128x64x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x16x!tt.ptr<f16>, #blocked1>
    %9 = tt.broadcast %7 : tensor<64x1xi32, #blocked1> -> tensor<64x16xi32, #blocked1>
    %10 = tt.addptr %8, %9 : tensor<64x16x!tt.ptr<f16>, #blocked1>, tensor<64x16xi32, #blocked1>
    %11:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst, %arg4 = %cst_0) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %12 = tt.load %10 : tensor<64x16x!tt.ptr<f16>, #blocked1>
      %13 = ttg.convert_layout %5 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %14 = ttg.convert_layout %12 : tensor<64x16xf16, #blocked1> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %15 = tt.dot %13, %14, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %16 = arith.truncf %15 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %17 = ttg.convert_layout %16 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %18 = ttg.local_alloc %12 : (tensor<64x16xf16, #blocked1>) -> !ttg.memdesc<64x16xf16, #shared, #smem>
      %19 = ttg.memdesc_trans %18 {order = array<i32: 1, 0>} : !ttg.memdesc<64x16xf16, #shared, #smem> -> !ttg.memdesc<16x64xf16, #shared1, #smem>
      %20 = ttg.local_load %19 : !ttg.memdesc<16x64xf16, #shared1, #smem> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %17, %20, %arg4 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %15, %21 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
    }
    tt.return %11#0, %11#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @nested_loops(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<32> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<true> : tensor<32x32xi1, #blocked>
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_2 = arith.constant dense<320> : tensor<32x1xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c10_i32 = arith.constant 10 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %3 = arith.muli %2, %cst_2 : tensor<32x1xi32, #blocked>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %8 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %9 = arith.muli %arg4, %c32_i32 : i32
      %10 = tt.splat %9 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %11 = tt.splat %9 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = arith.addi %10, %0 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %13 = arith.addi %11, %1 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = tt.expand_dims %12 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %15 = tt.broadcast %14 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
      %16 = tt.addptr %6, %15 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %17 = tt.load %16 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %18 = tt.expand_dims %13 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
      %19 = arith.muli %18, %cst_2 : tensor<32x1xi32, #blocked>
      %20 = tt.addptr %7, %19 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
      %21 = tt.broadcast %20 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
      %22 = tt.addptr %8, %19 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
      %23 = tt.broadcast %22 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
      %24 = ttg.local_alloc : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
      %25 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %26 = tt.broadcast %25 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
      %27 = tt.addptr %21, %26 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %28 = ttg.memdesc_index %24[%c0_i32] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
      %29 = ttg.async_copy_global_to_local %27, %28 mask %cst_0 {contiguity = 4 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
      %30 = ttg.async_commit_group tokens %29
      %31 = arith.addi %0, %cst : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %32 = tt.expand_dims %31 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %33 = tt.broadcast %32 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
      %34 = tt.addptr %21, %33 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %35 = ttg.memdesc_index %24[%c1_i32] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
      %36 = ttg.async_copy_global_to_local %34, %35 mask %cst_0 {contiguity = 4 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
      %37 = ttg.async_commit_group tokens %36
      %38:6 = scf.for %arg5 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg6 = %c1_i32, %arg7 = %c-1_i32, %arg8 = %30, %arg9 = %37, %arg10 = %26, %arg11 = %33) -> (i32, i32, !ttg.async.token, !ttg.async.token, tensor<32x32xi32, #blocked>, tensor<32x32xi32, #blocked>)  : i32 {
        %40 = arith.cmpi slt, %arg5, %c8_i32 : i32
        %41 = arith.addi %arg7, %c1_i32 : i32
        %42 = arith.cmpi sge, %41, %c2_i32 : i32
        %43 = arith.select %42, %c0_i32, %41 : i32
        %44 = ttg.async_wait %arg8 {num = 1 : i32}
        %45 = ttg.memdesc_index %24[%43] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
        %46 = ttg.local_load %45 token %44 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %47 = ttg.convert_layout %17 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
        %48 = tt.dot %46, %47, %cst_1, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
        %49 = tt.addptr %23, %arg10 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %50 = ttg.convert_layout %48 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
        tt.store %49, %50 : tensor<32x32x!tt.ptr<f32>, #blocked>
        %51 = arith.addi %arg6, %c1_i32 : i32
        %52 = arith.cmpi sge, %51, %c2_i32 : i32
        %53 = arith.select %52, %c0_i32, %51 : i32
        %54 = arith.addi %arg5, %c2_i32 : i32
        %55 = arith.muli %54, %c32_i32 : i32
        %56 = tt.splat %55 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %57 = arith.addi %56, %0 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %58 = tt.expand_dims %57 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
        %59 = tt.broadcast %58 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
        %60 = tt.addptr %21, %59 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %61 = ttg.memdesc_index %24[%53] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
        %62 = tt.splat %40 : i1 -> tensor<32x32xi1, #blocked>
        %63 = ttg.async_copy_global_to_local %60, %61 mask %62 {contiguity = 4 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
        %64 = ttg.async_commit_group tokens %63
        scf.yield %53, %43, %arg9, %64, %arg11, %59 : i32, i32, !ttg.async.token, !ttg.async.token, tensor<32x32xi32, #blocked>, tensor<32x32xi32, #blocked>
      }
      %39 = ttg.async_wait {num = 0 : i32}
      ttg.local_dealloc %24 : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @indirect_load_shared_layout(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg3: tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<1> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x16xi64, #shared1, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable>
    %3 = arith.cmpi sgt, %arg1, %c0 : index
    %4 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable> -> !ttg.memdesc<16xi64, #shared1, #smem, mutable>
    %5 = tt.splat %3 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = ttg.async_copy_global_to_local %arg3, %4 mask %5 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>> -> <16xi64, #shared1, #smem, mutable>
    %7 = ttg.async_commit_group tokens %6
    %8 = arith.cmpi sgt, %arg1, %c1 : index
    %9 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    %10 = tt.splat %3 : i1 -> tensor<16x16xi1, #blocked1>
    %11 = ttg.async_copy_global_to_local %arg2, %9 mask %10 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared, #smem, mutable>
    %12 = ttg.async_commit_group tokens %11
    %13 = ttg.async_wait %7 {num = 1 : i32}
    %14 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable> -> !ttg.memdesc<16xi64, #shared1, #smem, mutable>
    %15 = ttg.local_load %14 token %13 : !ttg.memdesc<16xi64, #shared1, #smem, mutable> -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
    %17 = tt.broadcast %16 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
    %18 = arith.muli %arg0, %17 : tensor<16x16xi64, #blocked>
    %19 = tt.addptr %arg5, %18 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %20 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    %21 = tt.splat %3 : i1 -> tensor<16x16xi1, #blocked>
    %22 = ttg.async_copy_global_to_local %19, %20 mask %21 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared, #smem, mutable>
    %23 = ttg.async_commit_group tokens %22
    %24 = tt.addptr %arg3, %cst_0 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %25 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable> -> !ttg.memdesc<16xi64, #shared1, #smem, mutable>
    %26 = tt.splat %8 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
    %27 = ttg.async_copy_global_to_local %24, %25 mask %26 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>> -> <16xi64, #shared1, #smem, mutable>
    %28 = ttg.async_commit_group tokens %27
    %29:8 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %24, %arg10 = %c0_i32, %arg11 = %c0_i32, %arg12 = %12, %arg13 = %23, %arg14 = %28) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      %31 = arith.subi %arg1, %c2 : index
      %32 = arith.cmpi slt, %arg6, %31 : index
      %33 = arith.subi %arg1, %c1 : index
      %34 = arith.cmpi slt, %arg6, %33 : index
      %35 = ttg.async_wait %arg12, %arg13 {num = 1 : i32}
      %36 = ttg.memdesc_index %0[%arg11] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
      %37 = ttg.local_load %36 token %35 : !ttg.memdesc<16x16xf16, #shared, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %38 = ttg.memdesc_index %2[%arg11] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
      %39 = ttg.local_load %38 token %35 : !ttg.memdesc<16x16xf16, #shared, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %40 = tt.dot %37, %39, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %41 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %42 = arith.addi %arg11, %c1_i32 : i32
      %43 = arith.cmpi sge, %42, %c1_i32 : i32
      %44 = arith.select %43, %c0_i32, %42 : i32
      %45 = ttg.memdesc_index %0[%arg10] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
      %46 = tt.splat %34 : i1 -> tensor<16x16xi1, #blocked1>
      %47 = ttg.async_copy_global_to_local %41, %45 mask %46 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared, #smem, mutable>
      %48 = ttg.async_commit_group tokens %47
      %49 = ttg.async_wait %arg14 {num = 1 : i32}
      %50 = ttg.memdesc_index %1[%44] : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable> -> !ttg.memdesc<16xi64, #shared1, #smem, mutable>
      %51 = ttg.local_load %50 token %49 : !ttg.memdesc<16xi64, #shared1, #smem, mutable> -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %52 = tt.expand_dims %51 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %53 = tt.broadcast %52 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
      %54 = arith.muli %arg0, %53 : tensor<16x16xi64, #blocked>
      %55 = tt.addptr %arg5, %54 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %56 = ttg.memdesc_index %2[%arg10] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
      %57 = tt.splat %34 : i1 -> tensor<16x16xi1, #blocked>
      %58 = ttg.async_copy_global_to_local %55, %56 mask %57 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared, #smem, mutable>
      %59 = ttg.async_commit_group tokens %58
      %60 = tt.addptr %arg9, %cst_0 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %61 = arith.addi %arg10, %c1_i32 : i32
      %62 = arith.cmpi sge, %61, %c1_i32 : i32
      %63 = arith.select %62, %c0_i32, %61 : i32
      %64 = ttg.memdesc_index %1[%63] : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable> -> !ttg.memdesc<16xi64, #shared1, #smem, mutable>
      %65 = tt.splat %32 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
      %66 = ttg.async_copy_global_to_local %60, %64 mask %65 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>> -> <16xi64, #shared1, #smem, mutable>
      %67 = ttg.async_commit_group tokens %66
      scf.yield %40, %41, %60, %63, %44, %48, %59, %67 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.num_stages = 3 : i32}
    %30 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %2 : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable>
    tt.return %29#0 : tensor<16x16xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @kernel_yield_constant(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
    %c2_i32 = arith.constant 2 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = arith.addi %arg4, %c31_i32 : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %7 = arith.cmpi sgt, %2, %c0_i32 : i32
    %8 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked>
    %9 = arith.cmpi slt, %3, %8 : tensor<32x1xi32, #blocked>
    %10 = tt.broadcast %9 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %11 = ttg.memdesc_index %6[%c0_i32] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %12 = tt.splat %7 : i1 -> tensor<32x32xi1, #blocked>
    %13 = arith.andi %12, %10 : tensor<32x32xi1, #blocked>
    %14 = ttg.async_copy_global_to_local %4, %11 mask %13 other %cst_1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    %15 = ttg.async_commit_group tokens %14
    %16 = arith.cmpi sgt, %2, %c1_i32 : i32
    %17 = arith.muli %arg5, %c32_i32 : i32
    %18 = tt.splat %17 : i32 -> tensor<32x32xi32, #blocked>
    %19 = tt.addptr %4, %18 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %20 = arith.subi %arg4, %c32_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<32x1xi32, #blocked>
    %22 = arith.cmpi slt, %3, %21 : tensor<32x1xi32, #blocked>
    %23 = tt.broadcast %22 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %24 = ttg.memdesc_index %6[%c1_i32] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %25 = tt.splat %16 : i1 -> tensor<32x32xi1, #blocked>
    %26 = arith.andi %25, %23 : tensor<32x32xi1, #blocked>
    %27 = ttg.async_copy_global_to_local %19, %24 mask %26 other %cst_1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    %28 = ttg.async_commit_group tokens %27
    %29:5 = scf.for %arg7 = %c0_i32 to %2 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %c1_i32, %arg10 = %c-1_i32, %arg11 = %15, %arg12 = %28) -> (tensor<32x32xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %31 = arith.subi %2, %c2_i32 : i32
      %32 = arith.cmpi slt, %arg7, %31 : i32
      %33 = arith.addi %arg10, %c1_i32 : i32
      %34 = arith.cmpi sge, %33, %c2_i32 : i32
      %35 = arith.select %34, %c0_i32, %33 : i32
      %36 = ttg.async_wait %arg11 {num = 1 : i32}
      %37 = ttg.memdesc_index %6[%35] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
      %38 = ttg.local_load %37 token %36 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %39 = tt.dot %cst_2, %38, %arg8, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %40 = ttg.convert_layout %39 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %5, %40 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %41 = arith.addi %arg9, %c1_i32 : i32
      %42 = arith.cmpi sge, %41, %c2_i32 : i32
      %43 = arith.select %42, %c0_i32, %41 : i32
      %44 = arith.addi %arg7, %c2_i32 : i32
      %45 = arith.muli %44, %c32_i32 : i32
      %46 = arith.muli %45, %arg5 : i32
      %47 = tt.splat %46 : i32 -> tensor<32x32xi32, #blocked>
      %48 = tt.addptr %4, %47 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %49 = arith.subi %arg4, %45 : i32
      %50 = tt.splat %49 : i32 -> tensor<32x1xi32, #blocked>
      %51 = arith.cmpi slt, %3, %50 : tensor<32x1xi32, #blocked>
      %52 = tt.broadcast %51 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %53 = ttg.memdesc_index %6[%43] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
      %54 = tt.splat %32 : i1 -> tensor<32x32xi1, #blocked>
      %55 = arith.andi %54, %52 : tensor<32x32xi1, #blocked>
      %56 = ttg.async_copy_global_to_local %48, %53 mask %55 other %cst_1 : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
      %57 = ttg.async_commit_group tokens %56
      scf.yield %cst_0, %43, %35, %arg12, %57 : tensor<32x32xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token
    }
    %30 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %6 : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
    %c2048_i32 = arith.constant 2048 : i32
    %c1014752_i32 = arith.constant 1014752 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1016800_i32 = arith.constant 1016800 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1016800_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    %9 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %10 = arith.addi %9, %2 : tensor<1024xi32, #blocked>
    %11 = arith.cmpi slt, %10, %3 : tensor<1024xi32, #blocked>
    %12 = tt.addptr %4, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %13 = ttg.memdesc_index %7[%c0_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %14 = ttg.async_copy_global_to_local %12, %13 mask %11 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %15 = ttg.async_commit_group tokens %14
    %16 = tt.addptr %5, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %17 = ttg.memdesc_index %8[%c0_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %18 = ttg.async_copy_global_to_local %16, %17 mask %11 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %19 = ttg.async_commit_group tokens %18
    %20 = arith.addi %1, %c1024_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<1024xi32, #blocked>
    %22 = arith.addi %21, %2 : tensor<1024xi32, #blocked>
    %23 = arith.cmpi slt, %22, %3 : tensor<1024xi32, #blocked>
    %24 = tt.addptr %4, %22 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %25 = ttg.memdesc_index %7[%c1_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %26 = ttg.async_copy_global_to_local %24, %25 mask %23 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %27 = ttg.async_commit_group tokens %26
    %28 = tt.addptr %5, %22 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %29 = ttg.memdesc_index %8[%c1_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %30 = ttg.async_copy_global_to_local %28, %29 mask %23 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %31 = ttg.async_commit_group tokens %30
    %32:10 = scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32 iter_args(%arg5 = %c1_i32, %arg6 = %c-1_i32, %arg7 = %15, %arg8 = %27, %arg9 = %19, %arg10 = %31, %arg11 = %10, %arg12 = %22, %arg13 = %11, %arg14 = %23) -> (i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi1, #blocked>, tensor<1024xi1, #blocked>)  : i32 {
      %34 = arith.cmpi slt, %arg4, %c1014752_i32 : i32
      %35 = arith.addi %arg6, %c1_i32 : i32
      %36 = arith.cmpi sge, %35, %c2_i32 : i32
      %37 = arith.select %36, %c0_i32, %35 : i32
      %38 = ttg.async_wait %arg7, %arg9 {num = 2 : i32}
      %39 = ttg.memdesc_index %7[%37] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %40 = ttg.local_load %39 token %38 : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
      %41 = ttg.memdesc_index %8[%37] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %42 = ttg.local_load %41 token %38 : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
      %43 = arith.addf %40, %42 : tensor<1024xf32, #blocked>
      %44 = tt.addptr %6, %arg11 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %44, %43, %arg13 : tensor<1024x!tt.ptr<f32>, #blocked>
      %45 = arith.addi %arg5, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.select %46, %c0_i32, %45 : i32
      %48 = arith.addi %arg4, %c2048_i32 : i32
      %49 = arith.addi %1, %48 : i32
      %50 = tt.splat %49 : i32 -> tensor<1024xi32, #blocked>
      %51 = arith.addi %50, %2 : tensor<1024xi32, #blocked>
      %52 = arith.cmpi slt, %51, %3 : tensor<1024xi32, #blocked>
      %53 = tt.addptr %4, %51 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %54 = ttg.memdesc_index %7[%47] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %55 = tt.splat %34 : i1 -> tensor<1024xi1, #blocked>
      %56 = arith.andi %55, %52 : tensor<1024xi1, #blocked>
      %57 = ttg.async_copy_global_to_local %53, %54 mask %56 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
      %58 = ttg.async_commit_group tokens %57
      %59 = tt.addptr %5, %51 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %60 = ttg.memdesc_index %8[%47] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %61 = tt.splat %34 : i1 -> tensor<1024xi1, #blocked>
      %62 = arith.andi %61, %52 : tensor<1024xi1, #blocked>
      %63 = ttg.async_copy_global_to_local %59, %60 mask %62 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
      %64 = ttg.async_commit_group tokens %63
      scf.yield %47, %37, %arg8, %58, %arg10, %64, %arg12, %51, %arg14, %52 : i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi1, #blocked>, tensor<1024xi1, #blocked>
    } {tt.num_stages = 3 : i32}
    %33 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %8 : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    ttg.local_dealloc %7 : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
  tt.func public @nested_loops(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<true> : tensor<16x16xi1, #blocked>
    %c-1_i32 = arith.constant -1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %2 = arith.muli %1, %cst_1 : tensor<16x1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %7 = tt.broadcast %4 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %8 = tt.broadcast %6 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %9 = tt.addptr %7, %8 : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %10 = tt.load %9 : tensor<16x16x!tt.ptr<f32>, #blocked>
      %11 = ttg.local_alloc %10 : (tensor<16x16xf32, #blocked>) -> !ttg.memdesc<16x16xf32, #shared, #smem>
      %12 = ttg.memdesc_trans %11 {order = array<i32: 1, 0>} : !ttg.memdesc<16x16xf32, #shared, #smem> -> !ttg.memdesc<16x16xf32, #shared1, #smem>
      %13 = ttg.local_load %12 : !ttg.memdesc<16x16xf32, #shared1, #smem> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %14 = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable>
      %15 = ttg.memdesc_index %14[%c0_i32] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
      %16 = ttg.async_copy_global_to_local %9, %15 mask %cst {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f32>, #blocked> -> <16x16xf32, #shared, #smem, mutable>
      %17 = ttg.async_commit_group tokens %16
      %18 = ttg.memdesc_index %14[%c1_i32] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
      %19 = ttg.async_copy_global_to_local %9, %18 mask %cst {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f32>, #blocked> -> <16x16xf32, #shared, #smem, mutable>
      %20 = ttg.async_commit_group tokens %19
      %21:4 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %c1_i32, %arg4 = %c-1_i32, %arg5 = %17, %arg6 = %20) -> (i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
        %23 = arith.cmpi slt, %arg2, %c0_i32 : i32
        %24 = arith.addi %arg4, %c1_i32 : i32
        %25 = arith.cmpi sge, %24, %c2_i32 : i32
        %26 = arith.select %25, %c0_i32, %24 : i32
        %27 = ttg.async_wait %arg5 {num = 1 : i32}
        %28 = ttg.memdesc_index %14[%26] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
        %29 = ttg.local_load %28 token %27 : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %30 = tt.dot %29, %13, %cst_0, inputPrecision = tf32 : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x16xf32, #mma>
        %31 = ttg.convert_layout %30 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
        tt.store %9, %31 : tensor<16x16x!tt.ptr<f32>, #blocked>
        %32 = arith.addi %arg3, %c1_i32 : i32
        %33 = arith.cmpi sge, %32, %c2_i32 : i32
        %34 = arith.select %33, %c0_i32, %32 : i32
        %35 = ttg.memdesc_index %14[%34] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
        %36 = tt.splat %23 : i1 -> tensor<16x16xi1, #blocked>
        %37 = ttg.async_copy_global_to_local %9, %35 mask %36 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f32>, #blocked> -> <16x16xf32, #shared, #smem, mutable>
        %38 = ttg.async_commit_group tokens %37
        scf.yield %34, %26, %arg6, %38 : i32, i32, !ttg.async.token, !ttg.async.token
      }
      %22 = ttg.async_wait {num = 0 : i32}
      ttg.local_dealloc %14 : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable>
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [16, 1, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [1, 8, 1], order = [2, 0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [16, 2, 1], threadsPerWarp = [4, 1, 8], warpsPerCTA = [1, 1, 8], order = [1, 0, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [32, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func public @int4_matmul_ampere(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<16x256xf32, #mma> {
    %cst = arith.constant dense<true> : tensor<64x256xi1, #blocked>
    %cst_0 = arith.constant dense<true> : tensor<16x128xi1, #blocked1>
    %c14_i32 = arith.constant 14 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_1 = arith.constant dense<64> : tensor<64x256xi32, #blocked>
    %cst_2 = arith.constant dense<128> : tensor<16x128xi32, #blocked1>
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant dense<4> : tensor<64x256xi8, #blocked>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #mma>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %2 = tt.broadcast %1 : tensor<1x128xi32, #blocked1> -> tensor<16x128xi32, #blocked1>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x128x!tt.ptr<f16>, #blocked1>
    %4 = tt.addptr %3, %2 : tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<16x128xi32, #blocked1>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %7 = tt.broadcast %6 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %8 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x256x!tt.ptr<i8>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<64x256x!tt.ptr<i8>, #blocked>, tensor<64x256xi32, #blocked>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<2x64x256xi8, #shared1, #smem, mutable>
    %12 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %13 = ttg.async_copy_global_to_local %4, %12 mask %cst_0 {contiguity = 8 : i32} : tensor<16x128x!tt.ptr<f16>, #blocked1> -> <16x128xf16, #shared, #smem, mutable>
    %14 = ttg.async_commit_group tokens %13
    %15 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<2x64x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xi8, #shared1, #smem, mutable>
    %16 = ttg.async_copy_global_to_local %9, %15 mask %cst {contiguity = 16 : i32} : tensor<64x256x!tt.ptr<i8>, #blocked> -> <64x256xi8, #shared1, #smem, mutable>
    %17 = ttg.async_commit_group tokens %16
    %18 = tt.addptr %4, %cst_2 : tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<16x128xi32, #blocked1>
    %19 = tt.addptr %9, %cst_1 : tensor<64x256x!tt.ptr<i8>, #blocked>, tensor<64x256xi32, #blocked>
    %20 = ttg.memdesc_index %10[%c1_i32] : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %21 = ttg.async_copy_global_to_local %18, %20 mask %cst_0 {contiguity = 8 : i32} : tensor<16x128x!tt.ptr<f16>, #blocked1> -> <16x128xf16, #shared, #smem, mutable>
    %22 = ttg.async_commit_group tokens %21
    %23 = ttg.memdesc_index %11[%c1_i32] : !ttg.memdesc<2x64x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xi8, #shared1, #smem, mutable>
    %24 = ttg.async_copy_global_to_local %19, %23 mask %cst {contiguity = 16 : i32} : tensor<64x256x!tt.ptr<i8>, #blocked> -> <64x256xi8, #shared1, #smem, mutable>
    %25 = ttg.async_commit_group tokens %24
    %26:9 = scf.for %arg2 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg3 = %cst_4, %arg4 = %18, %arg5 = %19, %arg6 = %c1_i32, %arg7 = %c-1_i32, %arg8 = %14, %arg9 = %22, %arg10 = %17, %arg11 = %25) -> (tensor<16x256xf32, #mma>, tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<i8>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %28 = arith.cmpi slt, %arg2, %c14_i32 : i32
      %29 = arith.addi %arg7, %c1_i32 : i32
      %30 = arith.cmpi sge, %29, %c2_i32 : i32
      %31 = arith.select %30, %c0_i32, %29 : i32
      %32 = ttg.async_wait %arg8, %arg10 {num = 2 : i32}
      %33 = ttg.memdesc_index %10[%31] : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
      %34 = ttg.local_load %33 token %32 : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %35 = ttg.memdesc_index %11[%31] : !ttg.memdesc<2x64x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xi8, #shared1, #smem, mutable>
      %36 = ttg.local_load %35 token %32 : !ttg.memdesc<64x256xi8, #shared1, #smem, mutable> -> tensor<64x256xi8, #blocked>
      %37 = arith.shli %36, %cst_3 : tensor<64x256xi8, #blocked>
      %38 = arith.shrsi %37, %cst_3 : tensor<64x256xi8, #blocked>
      %39 = arith.shrsi %36, %cst_3 : tensor<64x256xi8, #blocked>
      %40 = arith.sitofp %38 : tensor<64x256xi8, #blocked> to tensor<64x256xf16, #blocked>
      %41 = arith.sitofp %39 : tensor<64x256xi8, #blocked> to tensor<64x256xf16, #blocked>
      %42 = tt.join %40, %41 : tensor<64x256xf16, #blocked> -> tensor<64x256x2xf16, #blocked2>
      %43 = tt.trans %42 {order = array<i32: 0, 2, 1>} : tensor<64x256x2xf16, #blocked2> -> tensor<64x2x256xf16, #blocked3>
      %44 = tt.reshape %43 : tensor<64x2x256xf16, #blocked3> -> tensor<128x256xf16, #blocked4>
      %45 = ttg.convert_layout %44 : tensor<128x256xf16, #blocked4> -> tensor<128x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %46 = tt.dot %34, %45, %arg3 : tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x256xf32, #mma>
      %47 = tt.addptr %arg4, %cst_2 : tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<16x128xi32, #blocked1>
      %48 = tt.addptr %arg5, %cst_1 : tensor<64x256x!tt.ptr<i8>, #blocked>, tensor<64x256xi32, #blocked>
      %49 = arith.addi %arg6, %c1_i32 : i32
      %50 = arith.cmpi sge, %49, %c2_i32 : i32
      %51 = arith.select %50, %c0_i32, %49 : i32
      %52 = ttg.memdesc_index %10[%51] : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
      %53 = tt.splat %28 : i1 -> tensor<16x128xi1, #blocked1>
      %54 = ttg.async_copy_global_to_local %47, %52 mask %53 {contiguity = 8 : i32} : tensor<16x128x!tt.ptr<f16>, #blocked1> -> <16x128xf16, #shared, #smem, mutable>
      %55 = ttg.async_commit_group tokens %54
      %56 = ttg.memdesc_index %11[%51] : !ttg.memdesc<2x64x256xi8, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xi8, #shared1, #smem, mutable>
      %57 = tt.splat %28 : i1 -> tensor<64x256xi1, #blocked>
      %58 = ttg.async_copy_global_to_local %48, %56 mask %57 {contiguity = 16 : i32} : tensor<64x256x!tt.ptr<i8>, #blocked> -> <64x256xi8, #shared1, #smem, mutable>
      %59 = ttg.async_commit_group tokens %58
      scf.yield %46, %47, %48, %51, %31, %arg9, %55, %arg11, %59 : tensor<16x256xf32, #mma>, tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<i8>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    %27 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %11 : !ttg.memdesc<2x64x256xi8, #shared1, #smem, mutable>
    ttg.local_dealloc %10 : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable>
    tt.return %26#0 : tensor<16x256xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @load_convert_layout(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg3: tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %cst = arith.constant dense<1> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant dense<2> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = arith.cmpi slt, %0, %cst_0 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<1x16xi64, #shared1, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable>
    %5 = arith.cmpi sgt, %arg1, %c0 : index
    %6 = ttg.memdesc_index %3[%c0_i32] : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable> -> !ttg.memdesc<16xi64, #shared1, #smem, mutable>
    %7 = tt.splat %5 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
    %8 = arith.andi %7, %1 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = ttg.async_copy_global_to_local %arg3, %6 mask %8 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>> -> <16xi64, #shared1, #smem, mutable>
    %10 = ttg.async_commit_group tokens %9
    %11 = arith.cmpi sgt, %arg1, %c1 : index
    %12 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    %13 = tt.splat %5 : i1 -> tensor<16x16xi1, #blocked1>
    %14 = ttg.async_copy_global_to_local %arg2, %12 mask %13 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared, #smem, mutable>
    %15 = ttg.async_commit_group tokens %14
    %16 = ttg.async_wait %10 {num = 1 : i32}
    %17 = ttg.memdesc_index %3[%c0_i32] : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable> -> !ttg.memdesc<16xi64, #shared1, #smem, mutable>
    %18 = ttg.local_load %17 token %16 : !ttg.memdesc<16xi64, #shared1, #smem, mutable> -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
    %20 = tt.broadcast %19 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
    %21 = arith.muli %arg0, %20 : tensor<16x16xi64, #blocked>
    %22 = tt.addptr %arg5, %21 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %23 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    %24 = tt.splat %5 : i1 -> tensor<16x16xi1, #blocked>
    %25 = ttg.async_copy_global_to_local %22, %23 mask %24 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared, #smem, mutable>
    %26 = ttg.async_commit_group tokens %25
    %27 = tt.addptr %arg3, %cst : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %28 = ttg.memdesc_index %3[%c0_i32] : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable> -> !ttg.memdesc<16xi64, #shared1, #smem, mutable>
    %29 = tt.splat %11 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
    %30 = arith.andi %29, %1 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
    %31 = ttg.async_copy_global_to_local %27, %28 mask %30 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>> -> <16xi64, #shared1, #smem, mutable>
    %32 = ttg.async_commit_group tokens %31
    %33:8 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst_1, %arg8 = %arg2, %arg9 = %27, %arg10 = %c0_i32, %arg11 = %c0_i32, %arg12 = %15, %arg13 = %26, %arg14 = %32) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      %35 = arith.subi %arg1, %c2 : index
      %36 = arith.cmpi slt, %arg6, %35 : index
      %37 = arith.subi %arg1, %c1 : index
      %38 = arith.cmpi slt, %arg6, %37 : index
      %39 = ttg.async_wait %arg12, %arg13 {num = 1 : i32}
      %40 = ttg.memdesc_index %2[%arg11] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
      %41 = ttg.local_load %40 token %39 : !ttg.memdesc<16x16xf16, #shared, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %42 = ttg.memdesc_index %4[%arg11] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
      %43 = ttg.local_load %42 token %39 : !ttg.memdesc<16x16xf16, #shared, #smem, mutable> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %44 = tt.dot %41, %43, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %45 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %46 = arith.addi %arg11, %c1_i32 : i32
      %47 = arith.cmpi sge, %46, %c1_i32 : i32
      %48 = arith.select %47, %c0_i32, %46 : i32
      %49 = ttg.memdesc_index %2[%arg10] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
      %50 = tt.splat %38 : i1 -> tensor<16x16xi1, #blocked1>
      %51 = ttg.async_copy_global_to_local %45, %49 mask %50 {contiguity = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1> -> <16x16xf16, #shared, #smem, mutable>
      %52 = ttg.async_commit_group tokens %51
      %53 = ttg.async_wait %arg14 {num = 1 : i32}
      %54 = ttg.memdesc_index %3[%48] : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable> -> !ttg.memdesc<16xi64, #shared1, #smem, mutable>
      %55 = ttg.local_load %54 token %53 : !ttg.memdesc<16xi64, #shared1, #smem, mutable> -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %56 = tt.expand_dims %55 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %57 = tt.broadcast %56 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
      %58 = arith.muli %arg0, %57 : tensor<16x16xi64, #blocked>
      %59 = tt.addptr %arg5, %58 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %60 = ttg.memdesc_index %4[%arg10] : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
      %61 = tt.splat %38 : i1 -> tensor<16x16xi1, #blocked>
      %62 = ttg.async_copy_global_to_local %59, %60 mask %61 {contiguity = 4 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked> -> <16x16xf16, #shared, #smem, mutable>
      %63 = ttg.async_commit_group tokens %62
      %64 = tt.addptr %arg9, %cst : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %65 = arith.addi %arg10, %c1_i32 : i32
      %66 = arith.cmpi sge, %65, %c1_i32 : i32
      %67 = arith.select %66, %c0_i32, %65 : i32
      %68 = ttg.memdesc_index %3[%67] : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable> -> !ttg.memdesc<16xi64, #shared1, #smem, mutable>
      %69 = tt.splat %36 : i1 -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
      %70 = arith.andi %69, %1 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
      %71 = ttg.async_copy_global_to_local %64, %68 mask %70 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>> -> <16xi64, #shared1, #smem, mutable>
      %72 = ttg.async_commit_group tokens %71
      scf.yield %44, %45, %64, %67, %48, %52, %63, %72 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.num_stages = 3 : i32}
    %34 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %4 : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable>
    ttg.local_dealloc %3 : !ttg.memdesc<1x16xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %2 : !ttg.memdesc<1x16x16xf16, #shared, #smem, mutable>
    tt.return %33#0 : tensor<16x16xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
  tt.func public @matmul_indirect_pipeline(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<true> : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %4 = tt.broadcast %2 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %5 = tt.broadcast %3 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %6 = arith.addi %4, %5 : tensor<32x32xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %6 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %9 = tt.load %8 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %6 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %12 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 0, parent = #blocked}>>
    %13 = tt.addptr %12, %0 : tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<1x32xi64, #shared, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<1x32xf32, #shared, #smem, mutable>
    %17 = ttg.memdesc_index %15[%c0_i32] : !ttg.memdesc<1x32xi64, #shared, #smem, mutable> -> !ttg.memdesc<32xi64, #shared, #smem, mutable>
    %18 = ttg.async_copy_global_to_local %13, %17 mask %cst : tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 0, parent = #blocked}>> -> <32xi64, #shared, #smem, mutable>
    %19 = ttg.async_commit_group tokens %18
    %20 = ttg.async_wait %19 {num = 0 : i32}
    %21 = ttg.memdesc_index %15[%c0_i32] : !ttg.memdesc<1x32xi64, #shared, #smem, mutable> -> !ttg.memdesc<32xi64, #shared, #smem, mutable>
    %22 = ttg.local_load %21 token %20 : !ttg.memdesc<32xi64, #shared, #smem, mutable> -> tensor<32xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %23 = tt.addptr %14, %22 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<32xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %24 = ttg.memdesc_index %16[%c0_i32] : !ttg.memdesc<1x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %25 = ttg.async_copy_global_to_local %23, %24 mask %cst : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>> -> <32xf32, #shared, #smem, mutable>
    %26 = ttg.async_commit_group tokens %25
    %27 = ttg.memdesc_index %15[%c0_i32] : !ttg.memdesc<1x32xi64, #shared, #smem, mutable> -> !ttg.memdesc<32xi64, #shared, #smem, mutable>
    %28 = ttg.async_copy_global_to_local %13, %27 mask %cst : tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 0, parent = #blocked}>> -> <32xi64, #shared, #smem, mutable>
    %29 = ttg.async_commit_group tokens %28
    %30:4 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %c0_i32, %arg6 = %c0_i32, %arg7 = %26, %arg8 = %29) -> (i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %32 = arith.cmpi slt, %arg4, %c0_i32 : i32
      %33 = arith.cmpi slt, %arg4, %c1_i32 : i32
      %34 = ttg.async_wait %arg7, %arg8 {num = 0 : i32}
      %35 = ttg.memdesc_index %16[%arg6] : !ttg.memdesc<1x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
      %36 = ttg.local_load %35 token %34 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %37 = tt.expand_dims %36 {axis = 0 : i32} : tensor<32xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xf32, #blocked>
      %38 = tt.broadcast %37 : tensor<1x32xf32, #blocked> -> tensor<32x32xf32, #blocked>
      %39 = arith.addf %9, %38 : tensor<32x32xf32, #blocked>
      %40 = ttg.convert_layout %9 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %41 = ttg.convert_layout %39 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %42 = tt.dot %40, %41, %cst_0, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %43 = ttg.convert_layout %42 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %11, %43 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %44 = arith.addi %arg6, %c1_i32 : i32
      %45 = arith.cmpi sge, %44, %c1_i32 : i32
      %46 = arith.select %45, %c0_i32, %44 : i32
      %47 = ttg.memdesc_index %15[%46] : !ttg.memdesc<1x32xi64, #shared, #smem, mutable> -> !ttg.memdesc<32xi64, #shared, #smem, mutable>
      %48 = ttg.local_load %47 token %34 : !ttg.memdesc<32xi64, #shared, #smem, mutable> -> tensor<32xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
      %49 = tt.addptr %14, %48 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<32xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
      %50 = ttg.memdesc_index %16[%arg5] : !ttg.memdesc<1x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
      %51 = tt.splat %33 : i1 -> tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>>
      %52 = ttg.async_copy_global_to_local %49, %50 mask %51 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>> -> <32xf32, #shared, #smem, mutable>
      %53 = ttg.async_commit_group tokens %52
      %54 = arith.addi %arg5, %c1_i32 : i32
      %55 = arith.cmpi sge, %54, %c1_i32 : i32
      %56 = arith.select %55, %c0_i32, %54 : i32
      %57 = ttg.memdesc_index %15[%56] : !ttg.memdesc<1x32xi64, #shared, #smem, mutable> -> !ttg.memdesc<32xi64, #shared, #smem, mutable>
      %58 = tt.splat %32 : i1 -> tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>>
      %59 = ttg.async_copy_global_to_local %13, %57 mask %58 : tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 0, parent = #blocked}>> -> <32xi64, #shared, #smem, mutable>
      %60 = ttg.async_commit_group tokens %59
      scf.yield %56, %46, %53, %60 : i32, i32, !ttg.async.token, !ttg.async.token
    } {tt.num_stages = 3 : i32}
    %31 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %16 : !ttg.memdesc<1x32xf32, #shared, #smem, mutable>
    ttg.local_dealloc %15 : !ttg.memdesc<1x32xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @dont_pipeline_128x1(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<true> : tensor<128x1xi1, #blocked>
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<-1.000000e+30> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x128x1xi32, #shared, #smem, mutable>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>, #blocked>
    %2 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<2x128x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<128x1xi32, #shared, #smem, mutable>
    %3 = ttg.async_copy_global_to_local %1, %2 mask %cst : tensor<128x1x!tt.ptr<i32>, #blocked> -> <128x1xi32, #shared, #smem, mutable>
    %4 = ttg.async_commit_group tokens %3
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>, #blocked>
    %6 = ttg.memdesc_index %0[%c1_i32] : !ttg.memdesc<2x128x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<128x1xi32, #shared, #smem, mutable>
    %7 = ttg.async_copy_global_to_local %5, %6 mask %cst : tensor<128x1x!tt.ptr<i32>, #blocked> -> <128x1xi32, #shared, #smem, mutable>
    %8 = ttg.async_commit_group tokens %7
    %9:5 = scf.for %arg1 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg2 = %cst_0, %arg3 = %c1_i32, %arg4 = %c-1_i32, %arg5 = %4, %arg6 = %8) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %11 = arith.cmpi slt, %arg1, %c0_i32 : i32
      %12 = arith.addi %arg4, %c1_i32 : i32
      %13 = arith.cmpi sge, %12, %c2_i32 : i32
      %14 = arith.select %13, %c0_i32, %12 : i32
      %15 = ttg.async_wait %arg5 {num = 1 : i32}
      %16 = ttg.memdesc_index %0[%14] : !ttg.memdesc<2x128x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<128x1xi32, #shared, #smem, mutable>
      %17 = ttg.local_load %16 token %15 : !ttg.memdesc<128x1xi32, #shared, #smem, mutable> -> tensor<128x1xi32, #mma>
      %18 = tt.broadcast %17 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
      %19 = arith.sitofp %18 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>
      %20 = "tt.reduce"(%19) <{axis = 1 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %30 = arith.maxnumf %arg7, %arg8 : f32
        tt.reduce.return %30 : f32
      }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %21 = arith.maxnumf %arg2, %20 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %22 = arith.addi %arg3, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.select %23, %c0_i32, %22 : i32
      %25 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>, #blocked>
      %26 = ttg.memdesc_index %0[%24] : !ttg.memdesc<2x128x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<128x1xi32, #shared, #smem, mutable>
      %27 = tt.splat %11 : i1 -> tensor<128x1xi1, #blocked>
      %28 = ttg.async_copy_global_to_local %25, %26 mask %27 : tensor<128x1x!tt.ptr<i32>, #blocked> -> <128x1xi32, #shared, #smem, mutable>
      %29 = ttg.async_commit_group tokens %28
      scf.yield %21, %24, %14, %arg6, %29 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i32, i32, !ttg.async.token, !ttg.async.token
    }
    %10 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<2x128x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @matmul_nested_ops(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: index) -> tensor<128x128xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked1> -> tensor<32x128xi32, #blocked1>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
    %10 = tt.load %9 : tensor<32x128x!tt.ptr<f16>, #blocked1>
    %11 = ttg.convert_layout %10 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %12 = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %13 = arith.cmpi slt, %arg0, %arg1 : index
    %14 = arith.cmpi slt, %arg0, %arg5 : index
    %15 = scf.if %14 -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
      %31 = tt.addptr %4, %cst : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      scf.yield %31 : tensor<128x32x!tt.ptr<f16>, #blocked>
    } else {
      scf.yield %4 : tensor<128x32x!tt.ptr<f16>, #blocked>
    }
    %16 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %17 = tt.splat %13 : i1 -> tensor<128x32xi1, #blocked>
    %18 = ttg.async_copy_global_to_local %15, %16 mask %17 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked> -> <128x32xf16, #shared, #smem, mutable>
    %19 = ttg.async_commit_group tokens %18
    %20 = arith.addi %arg0, %arg2 : index
    %21 = arith.cmpi slt, %20, %arg1 : index
    %22 = tt.addptr %15, %cst : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    %23 = arith.cmpi slt, %20, %arg5 : index
    %24 = scf.if %23 -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
      %31 = tt.addptr %22, %cst : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      scf.yield %31 : tensor<128x32x!tt.ptr<f16>, #blocked>
    } else {
      scf.yield %22 : tensor<128x32x!tt.ptr<f16>, #blocked>
    }
    %25 = ttg.memdesc_index %12[%c1_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %26 = tt.splat %21 : i1 -> tensor<128x32xi1, #blocked>
    %27 = ttg.async_copy_global_to_local %24, %25 mask %26 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked> -> <128x32xf16, #shared, #smem, mutable>
    %28 = ttg.async_commit_group tokens %27
    %29:6 = scf.for %arg6 = %arg0 to %arg1 step %arg2 iter_args(%arg7 = %cst_0, %arg8 = %c1_i32, %arg9 = %c-1_i32, %arg10 = %24, %arg11 = %19, %arg12 = %28) -> (tensor<128x128xf32, #mma>, i32, i32, tensor<128x32x!tt.ptr<f16>, #blocked>, !ttg.async.token, !ttg.async.token) {
      %31 = arith.muli %arg2, %c2 : index
      %32 = arith.subi %arg1, %31 : index
      %33 = arith.cmpi slt, %arg6, %32 : index
      %34 = tt.addptr %arg10, %cst : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %35 = arith.muli %arg2, %c2 : index
      %36 = arith.addi %arg6, %35 : index
      %37 = arith.cmpi slt, %36, %arg5 : index
      %38 = scf.if %37 -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
        %53 = tt.addptr %34, %cst : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
        scf.yield %53 : tensor<128x32x!tt.ptr<f16>, #blocked>
      } else {
        scf.yield %34 : tensor<128x32x!tt.ptr<f16>, #blocked>
      }
      %39 = arith.addi %arg9, %c1_i32 : i32
      %40 = arith.cmpi sge, %39, %c2_i32 : i32
      %41 = arith.select %40, %c0_i32, %39 : i32
      %42 = ttg.async_wait %arg11 {num = 1 : i32}
      %43 = ttg.memdesc_index %12[%41] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %44 = ttg.local_load %43 token %42 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %45 = tt.dot %44, %11, %arg7 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %46 = arith.addi %arg8, %c1_i32 : i32
      %47 = arith.cmpi sge, %46, %c2_i32 : i32
      %48 = arith.select %47, %c0_i32, %46 : i32
      %49 = ttg.memdesc_index %12[%48] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %50 = tt.splat %33 : i1 -> tensor<128x32xi1, #blocked>
      %51 = ttg.async_copy_global_to_local %38, %49 mask %50 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked> -> <128x32xf16, #shared, #smem, mutable>
      %52 = ttg.async_commit_group tokens %51
      scf.yield %45, %48, %41, %38, %arg12, %52 : tensor<128x128xf32, #mma>, i32, i32, tensor<128x32x!tt.ptr<f16>, #blocked>, !ttg.async.token, !ttg.async.token
    }
    %30 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %12 : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    tt.return %29#0 : tensor<128x128xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @masked_add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
    %c2048_i32 = arith.constant 2048 : i32
    %c1014752_i32 = arith.constant 1014752 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1016800_i32 = arith.constant 1016800 : i32
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1016800_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    %9 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %10 = arith.addi %9, %2 : tensor<1024xi32, #blocked>
    %11 = arith.cmpi slt, %10, %3 : tensor<1024xi32, #blocked>
    %12 = tt.addptr %4, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %13 = ttg.memdesc_index %7[%c0_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %14 = ttg.async_copy_global_to_local %12, %13 mask %11 other %cst {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %15 = ttg.async_commit_group tokens %14
    %16 = tt.addptr %5, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %17 = ttg.memdesc_index %8[%c0_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %18 = ttg.async_copy_global_to_local %16, %17 mask %11 other %cst {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %19 = ttg.async_commit_group tokens %18
    %20 = arith.addi %1, %c1024_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<1024xi32, #blocked>
    %22 = arith.addi %21, %2 : tensor<1024xi32, #blocked>
    %23 = arith.cmpi slt, %22, %3 : tensor<1024xi32, #blocked>
    %24 = tt.addptr %4, %22 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %25 = ttg.memdesc_index %7[%c1_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %26 = ttg.async_copy_global_to_local %24, %25 mask %23 other %cst {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %27 = ttg.async_commit_group tokens %26
    %28 = tt.addptr %5, %22 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %29 = ttg.memdesc_index %8[%c1_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %30 = ttg.async_copy_global_to_local %28, %29 mask %23 other %cst {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %31 = ttg.async_commit_group tokens %30
    %32:10 = scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32 iter_args(%arg5 = %c1_i32, %arg6 = %c-1_i32, %arg7 = %15, %arg8 = %27, %arg9 = %11, %arg10 = %23, %arg11 = %19, %arg12 = %31, %arg13 = %10, %arg14 = %22) -> (i32, i32, !ttg.async.token, !ttg.async.token, tensor<1024xi1, #blocked>, tensor<1024xi1, #blocked>, !ttg.async.token, !ttg.async.token, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>)  : i32 {
      %34 = arith.cmpi slt, %arg4, %c1014752_i32 : i32
      %35 = arith.addi %arg6, %c1_i32 : i32
      %36 = arith.cmpi sge, %35, %c2_i32 : i32
      %37 = arith.select %36, %c0_i32, %35 : i32
      %38 = ttg.async_wait %arg7, %arg11 {num = 2 : i32}
      %39 = ttg.memdesc_index %7[%37] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %40 = ttg.local_load %39 token %38 : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
      %41 = arith.select %arg9, %40, %cst : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
      %42 = ttg.memdesc_index %8[%37] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %43 = ttg.local_load %42 token %38 : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
      %44 = arith.select %arg9, %43, %cst : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
      %45 = arith.addf %41, %44 : tensor<1024xf32, #blocked>
      %46 = tt.addptr %6, %arg13 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %46, %45, %arg9 : tensor<1024x!tt.ptr<f32>, #blocked>
      %47 = arith.addi %arg5, %c1_i32 : i32
      %48 = arith.cmpi sge, %47, %c2_i32 : i32
      %49 = arith.select %48, %c0_i32, %47 : i32
      %50 = arith.addi %arg4, %c2048_i32 : i32
      %51 = arith.addi %1, %50 : i32
      %52 = tt.splat %51 : i32 -> tensor<1024xi32, #blocked>
      %53 = arith.addi %52, %2 : tensor<1024xi32, #blocked>
      %54 = arith.cmpi slt, %53, %3 : tensor<1024xi32, #blocked>
      %55 = tt.addptr %4, %53 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %56 = ttg.memdesc_index %7[%49] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %57 = tt.splat %34 : i1 -> tensor<1024xi1, #blocked>
      %58 = arith.andi %57, %54 : tensor<1024xi1, #blocked>
      %59 = ttg.async_copy_global_to_local %55, %56 mask %58 other %cst {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
      %60 = ttg.async_commit_group tokens %59
      %61 = tt.addptr %5, %53 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %62 = ttg.memdesc_index %8[%49] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %63 = tt.splat %34 : i1 -> tensor<1024xi1, #blocked>
      %64 = arith.andi %63, %54 : tensor<1024xi1, #blocked>
      %65 = ttg.async_copy_global_to_local %61, %62 mask %64 other %cst {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
      %66 = ttg.async_commit_group tokens %65
      scf.yield %49, %37, %arg8, %60, %arg10, %54, %arg12, %66, %arg14, %53 : i32, i32, !ttg.async.token, !ttg.async.token, tensor<1024xi1, #blocked>, tensor<1024xi1, #blocked>, !ttg.async.token, !ttg.async.token, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>
    } {tt.num_stages = 3 : i32}
    %33 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %8 : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    ttg.local_dealloc %7 : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @predicate_stage1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
    %c2048_i32 = arith.constant 2048 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1016800_i32 = arith.constant 1016800 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1016800_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    %9 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %10 = arith.addi %9, %2 : tensor<1024xi32, #blocked>
    %11 = arith.cmpi slt, %10, %3 : tensor<1024xi32, #blocked>
    %12 = tt.addptr %4, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %13 = ttg.memdesc_index %7[%c0_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %14 = ttg.async_copy_global_to_local %12, %13 mask %11 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %15 = ttg.async_commit_group tokens %14
    %16 = tt.addptr %5, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %17 = ttg.memdesc_index %8[%c0_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %18 = ttg.async_copy_global_to_local %16, %17 mask %11 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %19 = ttg.async_commit_group tokens %18
    %20 = arith.addi %1, %c1024_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<1024xi32, #blocked>
    %22 = arith.addi %21, %2 : tensor<1024xi32, #blocked>
    %23 = arith.cmpi slt, %22, %3 : tensor<1024xi32, #blocked>
    %24 = tt.addptr %4, %22 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %25 = ttg.memdesc_index %7[%c1_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %26 = ttg.async_copy_global_to_local %24, %25 mask %23 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %27 = ttg.async_commit_group tokens %26
    %28 = tt.addptr %5, %22 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %29 = ttg.memdesc_index %8[%c1_i32] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
    %30 = ttg.async_copy_global_to_local %28, %29 mask %23 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
    %31 = ttg.async_commit_group tokens %30
    %32:10 = scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32 iter_args(%arg5 = %c1_i32, %arg6 = %c-1_i32, %arg7 = %15, %arg8 = %27, %arg9 = %19, %arg10 = %31, %arg11 = %10, %arg12 = %22, %arg13 = %11, %arg14 = %23) -> (i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi1, #blocked>, tensor<1024xi1, #blocked>)  : i32 {
      %34 = ttg.predicate_stage %arg4, %c1016800_i32, %c1024_i32 maxStage 2 stage 0 : i32 -> i1
      %35 = arith.addi %arg6, %c1_i32 : i32
      %36 = arith.cmpi sge, %35, %c2_i32 : i32
      %37 = arith.select %36, %c0_i32, %35 : i32
      %38 = ttg.async_wait %arg7, %arg9 {num = 2 : i32}
      %39 = ttg.memdesc_index %7[%37] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %40 = ttg.local_load %39 token %38 : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
      %41 = ttg.memdesc_index %8[%37] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %42 = ttg.local_load %41 token %38 : !ttg.memdesc<1024xf32, #shared, #smem, mutable> -> tensor<1024xf32, #blocked>
      %43 = arith.addf %40, %42 : tensor<1024xf32, #blocked>
      %44 = tt.addptr %6, %arg11 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %44, %43, %arg13 : tensor<1024x!tt.ptr<f32>, #blocked>
      %45 = arith.addi %arg5, %c1_i32 : i32
      %46 = arith.cmpi sge, %45, %c2_i32 : i32
      %47 = arith.select %46, %c0_i32, %45 : i32
      %48 = arith.addi %arg4, %c2048_i32 : i32
      %49 = arith.addi %1, %48 : i32
      %50 = tt.splat %49 : i32 -> tensor<1024xi32, #blocked>
      %51 = arith.addi %50, %2 : tensor<1024xi32, #blocked>
      %52 = arith.cmpi slt, %51, %3 : tensor<1024xi32, #blocked>
      %53 = tt.addptr %4, %51 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %54 = ttg.memdesc_index %7[%47] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %55 = tt.splat %34 : i1 -> tensor<1024xi1, #blocked>
      %56 = arith.andi %55, %52 : tensor<1024xi1, #blocked>
      %57 = ttg.async_copy_global_to_local %53, %54 mask %56 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
      %58 = ttg.async_commit_group tokens %57
      %59 = tt.addptr %5, %51 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %60 = ttg.memdesc_index %8[%47] : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable> -> !ttg.memdesc<1024xf32, #shared, #smem, mutable>
      %61 = tt.splat %34 : i1 -> tensor<1024xi1, #blocked>
      %62 = arith.andi %61, %52 : tensor<1024xi1, #blocked>
      %63 = ttg.async_copy_global_to_local %59, %60 mask %62 {contiguity = 4 : i32} : tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
      %64 = ttg.async_commit_group tokens %63
      scf.yield %47, %37, %arg8, %58, %arg10, %64, %arg12, %51, %arg14, %52 : i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi1, #blocked>, tensor<1024xi1, #blocked>
    } {__test_keep_predicate_stage, tt.num_stages = 3 : i32}
    %33 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %8 : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    ttg.local_dealloc %7 : !ttg.memdesc<2x1024xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @peeled_prologue_statically_dead(%arg0: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg1: tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<false> : tensor<128x32xi1, #blocked>
    %cst_0 = arith.constant dense<true> : tensor<128x32xi1, #blocked>
    %c3_i32 = arith.constant 3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %2 = ttg.async_copy_global_to_local %arg0, %1 mask %cst_0 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked> -> <128x32xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.memdesc_index %0[%c1_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %5 = ttg.async_copy_global_to_local %arg0, %4 mask %cst_0 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked> -> <128x32xf16, #shared, #smem, mutable>
    %6 = ttg.async_commit_group tokens %5
    %7 = ttg.memdesc_index %0[%c2_i32] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %8 = ttg.async_copy_global_to_local %arg0, %7 mask %cst {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked> -> <128x32xf16, #shared, #smem, mutable>
    %9 = ttg.async_commit_group tokens %8
    %10:6 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %c2_i32, %arg5 = %c-1_i32, %arg6 = %3, %arg7 = %6, %arg8 = %9) -> (tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %12 = arith.cmpi slt, %arg2, %c-1_i32 : i32
      %13 = arith.addi %arg5, %c1_i32 : i32
      %14 = arith.cmpi sge, %13, %c3_i32 : i32
      %15 = arith.select %14, %c0_i32, %13 : i32
      %16 = ttg.async_wait %arg6 {num = 2 : i32}
      %17 = ttg.memdesc_index %0[%15] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %18 = ttg.local_load %17 token %16 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %19 = tt.dot %18, %arg1, %arg3 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %20 = arith.addi %arg4, %c1_i32 : i32
      %21 = arith.cmpi sge, %20, %c3_i32 : i32
      %22 = arith.select %21, %c0_i32, %20 : i32
      %23 = ttg.memdesc_index %0[%22] : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %24 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked>
      %25 = ttg.async_copy_global_to_local %arg0, %23 mask %24 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked> -> <128x32xf16, #shared, #smem, mutable>
      %26 = ttg.async_commit_group tokens %25
      scf.yield %19, %22, %15, %arg7, %arg8, %26 : tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.num_stages = 4 : i32}
    %11 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<3x128x32xf16, #shared, #smem, mutable>
    tt.return %10#0 : tensor<128x128xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @barrier_in_loop_kernel(%arg0: tensor<1024x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1024_i32  : i32 {
      %0 = tt.load %arg0 : tensor<1024x!tt.ptr<f32>, #blocked>
      ttg.barrier local
      tt.store %arg0, %0 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @assert_in_loop_kernel(%arg0: tensor<1024x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg2: i1) {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg3 = %c0_i32 to %arg1 step %c1024_i32  : i32 {
      %0 = tt.load %arg0 : tensor<1024x!tt.ptr<f32>, #blocked>
      tt.assert %arg2, "some assert" : i1
      tt.store %arg0, %0 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @print_in_loop_kernel(%arg0: tensor<1024x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg2: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg3 = %c0_i32 to %arg1 step %c1024_i32  : i32 {
      %0 = tt.load %arg0 : tensor<1024x!tt.ptr<f32>, #blocked>
      tt.print "some print" {hex = false, isSigned = array<i32: 0>} : %arg2 : i32
      tt.store %arg0, %0 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
}

