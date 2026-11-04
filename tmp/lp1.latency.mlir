#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @matmul_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
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
    %cst = arith.constant dense<true> : tensor<128x32xi1, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked>
    %cst_1 = arith.constant dense<true> : tensor<32x128xi1, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_4 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_5 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %cst_6 = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %10:3 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %4, %arg7 = %9, %arg8 = %cst_3) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %11 = tt.load %arg6 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %12 = ttg.convert_layout %11 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %13 = tt.load %arg7, %cst_1, %cst_2 {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %14 = ttg.convert_layout %13 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %15 = arith.mulf %14, %cst_6 : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %16 = tt.dot %12, %15, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %17 = tt.addptr %arg6, %cst_4 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %18 = tt.addptr %arg7, %cst_5 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %17, %18, %16 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    }
    tt.return %10#2 : tensor<128x128xf32, #mma>
  }
  tt.func @matmul_loop_nested(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %cst) -> (tensor<128x128xf32, #mma>) {
      %1 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked>
      %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %4 = tt.broadcast %3 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
      %5 = tt.addptr %1, %4 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %6 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
      %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
      %9 = tt.broadcast %8 : tensor<1x128xi32, #blocked1> -> tensor<32x128xi32, #blocked1>
      %10 = tt.addptr %6, %9 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      %cst_0 = arith.constant dense<true> : tensor<128x32xi1, #blocked>
      %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked>
      %cst_2 = arith.constant dense<true> : tensor<32x128xi1, #blocked1>
      %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked1>
      %cst_4 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
      %cst_5 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
      %11:3 = scf.for %arg7 = %arg0 to %arg1 step %arg2 iter_args(%arg8 = %5, %arg9 = %10, %arg10 = %arg6) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
        %12 = tt.load %arg8, %cst_0, %cst_1 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
        %13 = ttg.convert_layout %12 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
        %14 = tt.load %arg9, %cst_2, %cst_3 {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
        %15 = ttg.convert_layout %14 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
        %16 = tt.dot %13, %15, %arg10 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
        %17 = tt.addptr %arg8, %cst_4 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
        %18 = tt.addptr %arg9, %cst_5 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
        scf.yield %17, %18, %16 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
      }
      scf.yield %11#2 : tensor<128x128xf32, #mma>
    }
    tt.return %0 : tensor<128x128xf32, #mma>
  }
  tt.func @matmul_loop_single_pipeline(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
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
    %cst = arith.constant dense<true> : tensor<128x32xi1, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked>
    %10 = tt.load %4, %cst, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>
    %11 = ttg.convert_layout %10 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<true> : tensor<32x128xi1, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_4 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %12:2 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %9, %arg7 = %cst_3) -> (tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %13 = tt.load %arg6, %cst_1, %cst_2 {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %14 = ttg.convert_layout %13 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %15 = tt.dot %11, %14, %arg7 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %16 = tt.addptr %arg6, %cst_4 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %16, %15 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    }
    tt.return %12#1 : tensor<128x128xf32, #mma>
  }
  tt.func @indirect_bmm_scalar(%arg0: i64 {tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg3: !tt.ptr<i64>, %arg4: tensor<16x16xi32, #blocked> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<16x16xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %0:3 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %arg3) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked>, !tt.ptr<i64>) {
      %1 = tt.load %arg8 {tt.latency = 1 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked>
      %2 = tt.load %arg9 {tt.latency = 1 : i32} : !tt.ptr<i64>
      %3 = arith.muli %arg0, %2 : i64
      %4 = tt.splat %3 : i64 -> tensor<16x16xi64, #blocked1>
      %5 = tt.addptr %arg5, %4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi64, #blocked1>
      %6 = tt.load %5 {tt.latency = 1 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %7 = ttg.convert_layout %1 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %8 = ttg.convert_layout %6 : tensor<16x16xf16, #blocked1> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %9 = tt.dot %7, %8, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %10 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
      %11 = tt.addptr %arg9, %c1_i32 : !tt.ptr<i64>, i32
      scf.yield %9, %10, %11 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked>, !tt.ptr<i64>
    } {tt.num_stages = 3 : i32}
    tt.return %0#0 : tensor<16x16xf32, #mma>
  }
  tt.func @indirect_bmm_scalar_dist_one(%arg0: i64 {tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg3: !tt.ptr<i64>, %arg4: tensor<16x16xi32, #blocked> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<16x16xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.load %arg3 : !tt.ptr<i64>
    %1 = tt.addptr %arg3, %c1_i32 : !tt.ptr<i64>, i32
    %2:4 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %1, %arg10 = %0) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked>, !tt.ptr<i64>, i64) {
      %3 = tt.load %arg8 {tt.latency = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked>
      %4 = tt.load %arg9 : !tt.ptr<i64>
      %5 = arith.muli %arg0, %arg10 : i64
      %6 = tt.splat %5 : i64 -> tensor<16x16xi64, #blocked1>
      %7 = tt.addptr %arg5, %6 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi64, #blocked1>
      %8 = tt.load %7 {tt.latency = 2 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %9 = ttg.convert_layout %3 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %10 = ttg.convert_layout %8 : tensor<16x16xf16, #blocked1> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %11 = tt.dot %9, %10, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %12 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
      %13 = tt.addptr %arg9, %c1_i32 : !tt.ptr<i64>, i32
      scf.yield %11, %12, %13, %4 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked>, !tt.ptr<i64>, i64
    }
    tt.return %2#0 : tensor<16x16xf32, #mma>
  }
  tt.func @indirect_bmm_vector(%arg0: tensor<16x16xi64, #blocked1> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg3: tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked1}>>, %arg4: tensor<16x16xi32, #blocked> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<16x16xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.splat %c1_i32 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %1:3 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %arg3) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked1}>>) {
      %2 = tt.load %arg8 {tt.latency = 1 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked>
      %3 = tt.load %arg9 {tt.latency = 1 : i32} : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi64, #blocked1>
      %5 = tt.broadcast %4 : tensor<16x1xi64, #blocked1> -> tensor<16x16xi64, #blocked1>
      %6 = arith.muli %arg0, %5 : tensor<16x16xi64, #blocked1>
      %7 = tt.addptr %arg5, %6 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi64, #blocked1>
      %8 = tt.load %7 {tt.latency = 1 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %9 = ttg.convert_layout %2 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %10 = ttg.convert_layout %8 : tensor<16x16xf16, #blocked1> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %11 = tt.dot %9, %10, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %12 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
      %13 = tt.addptr %arg9, %0 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      scf.yield %11, %12, %13 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked1}>>
    } {tt.num_stages = 3 : i32}
    tt.return %1#0 : tensor<16x16xf32, #mma>
  }
  tt.func @post_load_inv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c900 = arith.constant 900 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %3 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked>
    %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %5 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %6:3 = scf.for %arg9 = %c0 to %c900 step %c1 iter_args(%arg10 = %cst, %arg11 = %1, %arg12 = %2) -> (tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>) {
      %7 = arith.index_cast %arg9 : index to i32
      %8 = arith.muli %7, %c32_i32 : i32
      %9 = arith.subi %arg5, %8 : i32
      %10 = tt.splat %9 : i32 -> tensor<1x32xi32, #blocked>
      %11 = arith.cmpi slt, %0, %10 : tensor<1x32xi32, #blocked>
      %12 = tt.broadcast %11 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %13 = tt.load %arg11, %12, %cst_0 {tt.latency = 2 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked>
      %14 = tt.splat %9 : i32 -> tensor<32x1xi32, #blocked>
      %15 = arith.cmpi slt, %3, %14 : tensor<32x1xi32, #blocked>
      %16 = tt.broadcast %15 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %17 = tt.load %arg12, %16, %cst_0 {tt.latency = 2 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked>
      %18 = ttg.convert_layout %13 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %19 = ttg.convert_layout %17 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %20 = tt.dot %18, %19, %arg10, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %21 = arith.index_cast %arg9 : index to i32
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.muli %22, %c32_i32 : i32
      %24 = tt.splat %23 : i32 -> tensor<32x32xi32, #blocked>
      %25 = tt.addptr %4, %24 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %26 = arith.muli %23, %arg7 : i32
      %27 = tt.splat %26 : i32 -> tensor<32x32xi32, #blocked>
      %28 = tt.addptr %5, %27 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %20, %25, %28 : tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>
    }
    tt.return %6#0 : tensor<32x32xf32, #mma>
  }
  tt.func @cross_iter_dep(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #blocked>
    %6 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked>
    %7 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %8:5 = scf.for %arg9 = %c0 to %c32 step %c1 iter_args(%arg10 = %cst, %arg11 = %0, %arg12 = %1, %arg13 = %3, %arg14 = %4) -> (tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>) {
      %9 = arith.index_cast %arg9 : index to i32
      %10 = arith.muli %9, %c32_i32 : i32
      %11 = arith.subi %arg5, %10 : i32
      %12 = tt.splat %11 : i32 -> tensor<1x32xi32, #blocked>
      %13 = arith.cmpi slt, %5, %12 : tensor<1x32xi32, #blocked>
      %14 = tt.broadcast %13 : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %15 = tt.load %arg11, %14, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %16 = tt.splat %11 : i32 -> tensor<32x1xi32, #blocked>
      %17 = arith.cmpi slt, %6, %16 : tensor<32x1xi32, #blocked>
      %18 = tt.broadcast %17 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %19 = tt.load %arg12, %18, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %20 = ttg.convert_layout %15 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %21 = ttg.convert_layout %19 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %22 = tt.dot %20, %21, %arg10, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %23 = arith.index_cast %arg9 : index to i32
      %24 = arith.addi %23, %c2_i32 : i32
      %25 = arith.muli %24, %c32_i32 : i32
      %26 = tt.splat %25 : i32 -> tensor<32x32xi32, #blocked>
      %27 = tt.addptr %7, %26 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %28 = arith.muli %25, %arg7 : i32
      %29 = tt.splat %28 : i32 -> tensor<32x32xi32, #blocked>
      %30 = tt.addptr %2, %29 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      scf.yield %22, %arg13, %arg14, %27, %30 : tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32x!tt.ptr<f32>, #blocked>
    }
    tt.return %8#0 : tensor<32x32xf32, #mma>
  }
  tt.func @dep_arg_two_uses(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c100 = arith.constant 100 : index
    %c64_i64 = arith.constant 64 : i64
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %4 = tt.splat %c64_i64 : i64 -> tensor<1x32xi64, #blocked>
    %5 = tt.splat %c64_i64 : i64 -> tensor<1x32xi64, #blocked>
    %6 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked>
    %c32 = arith.constant 32 : index
    %7 = arith.index_cast %c32 : index to i32
    %8 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #blocked1>
    %9 = arith.truncf %cst : tensor<32x128xf32, #blocked1> to tensor<32x128xf16, #blocked1>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %10 = tt.splat %c64_i64 : i64 -> tensor<32x128xi64, #blocked1>
    %11 = tt.addptr %arg1, %7 : !tt.ptr<i32>, i32
    %c0 = arith.constant 0 : index
    %12:5 = scf.for %arg3 = %c0 to %c100 step %c32 iter_args(%arg4 = %6, %arg5 = %3, %arg6 = %11, %arg7 = %cst_0, %arg8 = %8) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, !tt.ptr<i32>, tensor<128x128xf32, #mma>, tensor<32x128x!tt.ptr<f16>, #blocked1>) {
      %13 = arith.subi %c100, %arg3 : index
      %14 = arith.index_cast %13 : index to i32
      %15 = tt.splat %14 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %16 = tt.splat %14 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %17 = arith.cmpi slt, %1, %15 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %18 = arith.cmpi slt, %2, %16 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %19 = tt.expand_dims %17 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi1, #blocked>
      %20 = tt.expand_dims %18 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi1, #blocked1>
      %21 = tt.expand_dims %arg5 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %22 = tt.expand_dims %arg5 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %23 = arith.extsi %21 : tensor<1x32xi32, #blocked> to tensor<1x32xi64, #blocked>
      %24 = arith.extsi %22 : tensor<1x32xi32, #blocked> to tensor<1x32xi64, #blocked>
      %25 = arith.muli %23, %4 : tensor<1x32xi64, #blocked>
      %26 = arith.muli %24, %5 : tensor<1x32xi64, #blocked>
      %27 = tt.broadcast %25 : tensor<1x32xi64, #blocked> -> tensor<128x32xi64, #blocked>
      %28 = tt.broadcast %26 : tensor<1x32xi64, #blocked> -> tensor<128x32xi64, #blocked>
      %29 = tt.addptr %arg4, %27 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi64, #blocked>
      %30 = tt.addptr %arg4, %28 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi64, #blocked>
      %31 = tt.broadcast %19 : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>
      %32 = tt.load %30, %31 : tensor<128x32x!tt.ptr<f16>, #blocked>
      %33 = tt.splat %arg6 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>
      %34 = tt.addptr %33, %0 : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %35 = tt.load %34 : tensor<32x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>
      %36 = tt.addptr %arg6, %7 : !tt.ptr<i32>, i32
      %37 = tt.broadcast %20 : tensor<32x1xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
      %38 = tt.load %arg8, %37, %9 : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %39 = ttg.convert_layout %32 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %40 = ttg.convert_layout %38 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %41 = tt.dot %39, %40, %arg7 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %42 = tt.addptr %arg8, %10 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi64, #blocked1>
      scf.yield %29, %35, %36, %41, %42 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, !tt.ptr<i32>, tensor<128x128xf32, #mma>, tensor<32x128x!tt.ptr<f16>, #blocked1>
    }
    tt.return %12#3 : tensor<128x128xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @load_two_users_incompatible_layouts(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_0 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %17:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = ttg.convert_layout %9 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = ttg.convert_layout %18 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %19, %20, %cst_1 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %23 = ttg.convert_layout %22 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %24 = ttg.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared, #smem>
      %25 = ttg.memdesc_trans %24 {order = array<i32: 1, 0>} : !ttg.memdesc<64x16xf16, #shared, #smem> -> !ttg.memdesc<16x64xf16, #shared1, #smem>
      %26 = ttg.local_load %25 : !ttg.memdesc<16x64xf16, #shared1, #smem> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %23, %26, %arg4 : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %21, %27 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
    }
    tt.return %17#0, %17#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @nested_loops(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<320> : tensor<32x1xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c10_i32 = arith.constant 10 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %3 = arith.muli %2, %cst_0 : tensor<32x1xi32, #blocked>
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
      %19 = arith.muli %18, %cst_0 : tensor<32x1xi32, #blocked>
      %20 = tt.addptr %7, %19 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
      %21 = tt.broadcast %20 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
      %22 = tt.addptr %8, %19 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
      %23 = tt.broadcast %22 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
      scf.for %arg5 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
        %24 = arith.muli %arg5, %c32_i32 : i32
        %25 = tt.splat %24 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %26 = arith.addi %25, %0 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %27 = tt.expand_dims %26 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
        %28 = tt.broadcast %27 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
        %29 = tt.addptr %21, %28 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %30 = tt.load %29 {tt.latency = 2 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked>
        %31 = ttg.convert_layout %30 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %32 = ttg.convert_layout %17 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
        %33 = tt.dot %31, %32, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
        %34 = tt.addptr %23, %28 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %35 = ttg.convert_layout %33 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
        tt.store %34, %35 : tensor<32x32x!tt.ptr<f32>, #blocked>
      }
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @indirect_load_shared_layout(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg3: tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<16x16xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.splat %c1_i32 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1:3 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %arg3) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>) {
      %2 = tt.load %arg8 {tt.latency = 1 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %3 = tt.load %arg9 {tt.latency = 1 : i32} : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
      %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %5 = tt.broadcast %4 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
      %6 = arith.muli %arg0, %5 : tensor<16x16xi64, #blocked>
      %7 = tt.addptr %arg5, %6 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %8 = tt.load %7 {tt.latency = 1 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked>
      %9 = ttg.convert_layout %2 : tensor<16x16xf16, #blocked1> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %10 = ttg.convert_layout %8 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %11 = tt.dot %9, %10, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %12 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %13 = tt.addptr %arg9, %0 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %11, %12, %13 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
    } {tt.num_stages = 3 : i32}
    tt.return %1#0 : tensor<16x16xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @kernel_yield_constant(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = arith.addi %arg4, %c31_i32 : i32
    %3 = arith.divsi %2, %c32_i32 : i32
    %4 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %7 = scf.for %arg7 = %c0_i32 to %3 step %c1_i32 iter_args(%arg8 = %cst) -> (tensor<32x32xf32, #mma>)  : i32 {
      %8 = arith.muli %arg7, %c32_i32 : i32
      %9 = arith.muli %8, %arg5 : i32
      %10 = tt.splat %9 : i32 -> tensor<32x32xi32, #blocked>
      %11 = tt.addptr %5, %10 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %12 = arith.subi %arg4, %8 : i32
      %13 = tt.splat %12 : i32 -> tensor<32x1xi32, #blocked>
      %14 = arith.cmpi slt, %4, %13 : tensor<32x1xi32, #blocked>
      %15 = tt.broadcast %14 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %16 = tt.load %11, %15, %cst_1 {tt.latency = 2 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked>
      %17 = ttg.convert_layout %16 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %18 = tt.dot %cst_2, %17, %arg8, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %19 = ttg.convert_layout %18 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %6, %19 : tensor<32x32x!tt.ptr<f32>, #blocked>
      scf.yield %cst_0 : tensor<32x32xf32, #mma>
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
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
    scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32  : i32 {
      %7 = arith.addi %1, %arg4 : i32
      %8 = tt.splat %7 : i32 -> tensor<1024xi32, #blocked>
      %9 = arith.addi %8, %2 : tensor<1024xi32, #blocked>
      %10 = arith.cmpi slt, %9, %3 : tensor<1024xi32, #blocked>
      %11 = tt.addptr %4, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %12 = tt.load %11, %10 {tt.latency = 2 : i32} : tensor<1024x!tt.ptr<f32>, #blocked>
      %13 = tt.addptr %5, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %14 = tt.load %13, %10 {tt.latency = 2 : i32} : tensor<1024x!tt.ptr<f32>, #blocked>
      %15 = arith.addf %12, %14 : tensor<1024xf32, #blocked>
      %16 = tt.addptr %6, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %16, %15, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
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
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %2 = arith.muli %1, %cst_0 : tensor<16x1xi32, #blocked>
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
      scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %14 = tt.load %9 {tt.latency = 2 : i32} : tensor<16x16x!tt.ptr<f32>, #blocked>
        %15 = ttg.convert_layout %14 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %16 = tt.dot %15, %13, %cst, inputPrecision = tf32 : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x16xf32, #mma>
        %17 = ttg.convert_layout %16 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
        tt.store %9, %17 : tensor<16x16x!tt.ptr<f32>, #blocked>
      }
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
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func public @int4_matmul_ampere(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<16x256xf32, #mma> {
    %cst = arith.constant dense<64> : tensor<64x256xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<16x128xi32, #blocked1>
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x128xf16, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c15_i32 = arith.constant 15 : i32
    %cst_2 = arith.constant dense<4> : tensor<64x256xi8, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #mma>
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
    %10:3 = scf.for %arg2 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg3 = %cst_3, %arg4 = %4, %arg5 = %9) -> (tensor<16x256xf32, #mma>, tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<i8>, #blocked>)  : i32 {
      %11 = tt.load %arg4 {tt.latency = 2 : i32} : tensor<16x128x!tt.ptr<f16>, #blocked1>
      %12 = tt.load %arg5 {tt.latency = 2 : i32} : tensor<64x256x!tt.ptr<i8>, #blocked>
      %13 = arith.shli %12, %cst_2 : tensor<64x256xi8, #blocked>
      %14 = arith.shrsi %13, %cst_2 : tensor<64x256xi8, #blocked>
      %15 = arith.shrsi %12, %cst_2 : tensor<64x256xi8, #blocked>
      %16 = arith.sitofp %14 : tensor<64x256xi8, #blocked> to tensor<64x256xf16, #blocked>
      %17 = arith.sitofp %15 : tensor<64x256xi8, #blocked> to tensor<64x256xf16, #blocked>
      %18 = tt.join %16, %17 : tensor<64x256xf16, #blocked> -> tensor<64x256x2xf16, #blocked2>
      %19 = tt.trans %18 {order = array<i32: 0, 2, 1>} : tensor<64x256x2xf16, #blocked2> -> tensor<64x2x256xf16, #blocked3>
      %20 = tt.reshape %19 : tensor<64x2x256xf16, #blocked3> -> tensor<128x256xf16, #blocked4>
      %21 = ttg.convert_layout %11 : tensor<16x128xf16, #blocked1> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %22 = ttg.convert_layout %20 : tensor<128x256xf16, #blocked4> -> tensor<128x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %23 = tt.dot %21, %22, %arg3 : tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x256xf32, #mma>
      %24 = tt.addptr %arg4, %cst_0 : tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<16x128xi32, #blocked1>
      %25 = tt.addptr %arg5, %cst : tensor<64x256x!tt.ptr<i8>, #blocked>, tensor<64x256xi32, #blocked>
      scf.yield %23, %24, %25 : tensor<16x256xf32, #mma>, tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<i8>, #blocked>
    }
    tt.return %10#0 : tensor<16x256xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @load_convert_layout(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg3: tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = dense<16> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<16x16xf32, #mma> {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %cst_0 = arith.constant dense<2> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %1 = tt.splat %c1_i32 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = arith.cmpi slt, %0, %cst_0 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3:3 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %arg3) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>) {
      %4 = tt.load %arg8 {tt.latency = 1 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %5 = tt.load %arg9, %2 {tt.latency = 1 : i32} : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
      %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<16xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %7 = tt.broadcast %6 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
      %8 = arith.muli %arg0, %7 : tensor<16x16xi64, #blocked>
      %9 = tt.addptr %arg5, %8 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %10 = tt.load %9 {tt.latency = 1 : i32} : tensor<16x16x!tt.ptr<f16>, #blocked>
      %11 = ttg.convert_layout %4 : tensor<16x16xf16, #blocked1> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %12 = ttg.convert_layout %10 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %13 = tt.dot %11, %12, %arg7 : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %14 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %15 = tt.addptr %arg9, %1 : tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %13, %14, %15 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
    } {tt.num_stages = 3 : i32}
    tt.return %3#0 : tensor<16x16xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
  tt.func public @matmul_indirect_pipeline(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
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
    scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %15 = tt.load %13 {tt.latency = 1 : i32} : tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 0, parent = #blocked}>>
      %16 = tt.addptr %14, %15 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<32xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
      %17 = tt.load %16 {tt.latency = 1 : i32} : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>
      %18 = tt.expand_dims %17 {axis = 0 : i32} : tensor<32xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xf32, #blocked>
      %19 = tt.broadcast %18 : tensor<1x32xf32, #blocked> -> tensor<32x32xf32, #blocked>
      %20 = arith.addf %9, %19 : tensor<32x32xf32, #blocked>
      %21 = ttg.convert_layout %9 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %22 = ttg.convert_layout %20 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %23 = tt.dot %21, %22, %cst, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %24 = ttg.convert_layout %23 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %11, %24 : tensor<32x32x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @dont_pipeline_128x1(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<-1.000000e+30> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %0 = scf.for %arg1 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg2 = %cst_0) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>)  : i32 {
      %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>, #blocked>
      %2 = tt.load %1 {tt.latency = 2 : i32} : tensor<128x1x!tt.ptr<i32>, #blocked>
      %3 = ttg.convert_layout %2 : tensor<128x1xi32, #blocked> -> tensor<128x1xi32, #mma>
      %4 = tt.broadcast %3 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
      %5 = arith.sitofp %4 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>
      %6 = "tt.reduce"(%5) <{axis = 1 : i32}> ({
      ^bb0(%arg3: f32, %arg4: f32):
        %11 = arith.maxnumf %arg3, %arg4 : f32
        tt.reduce.return %11 : f32
      }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %7 = arith.maxnumf %arg2, %6 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %8 = arith.truncf %5 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
      %9 = ttg.convert_layout %8 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
      %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %10 = tt.dot %9, %cst_2, %cst_1 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %7 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @matmul_nested_ops(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: index) -> tensor<128x128xf32, #mma> {
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
    %cst = arith.constant dense<true> : tensor<128x32xi1, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked>
    %cst_1 = arith.constant dense<true> : tensor<32x128xi1, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_4 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %10 = tt.load %9, %cst_1, %cst_2 : tensor<32x128x!tt.ptr<f16>, #blocked1>
    %11 = ttg.convert_layout %10 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %12:2 = scf.for %arg6 = %arg0 to %arg1 step %arg2 iter_args(%arg7 = %4, %arg8 = %cst_3) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>) {
      %13 = arith.cmpi slt, %arg6, %arg5 : index
      %14 = scf.if %13 -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
        %19 = tt.addptr %arg7, %cst_4 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
        scf.yield %19 : tensor<128x32x!tt.ptr<f16>, #blocked>
      } else {
        scf.yield %arg7 : tensor<128x32x!tt.ptr<f16>, #blocked>
      }
      %15 = tt.load %14 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %16 = ttg.convert_layout %15 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %17 = tt.dot %16, %11, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %18 = tt.addptr %14, %cst_4 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      scf.yield %18, %17 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>
    }
    tt.return %12#1 : tensor<128x128xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @masked_add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
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
    scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32  : i32 {
      %7 = arith.addi %1, %arg4 : i32
      %8 = tt.splat %7 : i32 -> tensor<1024xi32, #blocked>
      %9 = arith.addi %8, %2 : tensor<1024xi32, #blocked>
      %10 = arith.cmpi slt, %9, %3 : tensor<1024xi32, #blocked>
      %11 = tt.addptr %4, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %12 = tt.load %11, %10, %cst {tt.latency = 2 : i32} : tensor<1024x!tt.ptr<f32>, #blocked>
      %13 = tt.addptr %5, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %14 = tt.load %13, %10, %cst {tt.latency = 2 : i32} : tensor<1024x!tt.ptr<f32>, #blocked>
      %15 = arith.addf %12, %14 : tensor<1024xf32, #blocked>
      %16 = tt.addptr %6, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %16, %15, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @predicate_stage1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
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
    scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32  : i32 {
      %7 = arith.addi %1, %arg4 : i32
      %8 = tt.splat %7 : i32 -> tensor<1024xi32, #blocked>
      %9 = arith.addi %8, %2 : tensor<1024xi32, #blocked>
      %10 = arith.cmpi slt, %9, %3 : tensor<1024xi32, #blocked>
      %11 = tt.addptr %4, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %12 = tt.load %11, %10 {tt.latency = 2 : i32} : tensor<1024x!tt.ptr<f32>, #blocked>
      %13 = tt.addptr %5, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %14 = tt.load %13, %10 {tt.latency = 2 : i32} : tensor<1024x!tt.ptr<f32>, #blocked>
      %15 = arith.addf %12, %14 : tensor<1024xf32, #blocked>
      %16 = tt.addptr %6, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %16, %15, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {__test_keep_predicate_stage, tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @peeled_prologue_statically_dead(%arg0: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg1: tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>) -> tensor<128x128xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %cst) -> (tensor<128x128xf32, #mma>)  : i32 {
      %1 = tt.load %arg0 {tt.latency = 3 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %2 = ttg.convert_layout %1 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %3 = tt.dot %2, %arg1, %arg3 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      scf.yield %3 : tensor<128x128xf32, #mma>
    } {tt.num_stages = 4 : i32}
    tt.return %0 : tensor<128x128xf32, #mma>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @barrier_in_loop_kernel(%arg0: tensor<1024x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1024_i32  : i32 {
      %0 = tt.load %arg0 {tt.latency = 1 : i32} : tensor<1024x!tt.ptr<f32>, #blocked>
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
      %0 = tt.load %arg0 {tt.latency = 1 : i32} : tensor<1024x!tt.ptr<f32>, #blocked>
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
      %0 = tt.load %arg0 {tt.latency = 1 : i32} : tensor<1024x!tt.ptr<f32>, #blocked>
      tt.print "some print" {hex = false, isSigned = array<i32: 0>} : %arg2 : i32
      tt.store %arg0, %0 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
}

