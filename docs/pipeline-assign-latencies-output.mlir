  tt.func @default_stages(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<32x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %0:3 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %arg3, %arg7 = %arg4, %arg8 = %cst) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %1 = tt.load %arg6 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %2 = ttg.convert_layout %1 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %3 = tt.load %arg7 {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %4 = ttg.convert_layout %3 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %5 = tt.dot %2, %4, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %6 = tt.addptr %arg6, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %7 = tt.addptr %arg7, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %6, %7, %5 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    }
    tt.return %0#2 : tensor<128x128xf32, #mma>
  }
  tt.func @small_load(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<32x128x!tt.ptr<f16>, #blocked1>) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %0:3 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %arg3, %arg7 = %arg4, %arg8 = %cst) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %1 = tt.load %arg6 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %2 = ttg.convert_layout %1 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %3 = tt.load %arg7 : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %4 = ttg.convert_layout %3 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %5 = tt.dot %2, %4, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %6 = tt.addptr %arg6, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %7 = tt.addptr %arg7, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %6, %7, %5 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    }
    tt.return %0#2 : tensor<128x128xf32, #mma>
  }
  tt.func @load_into_shared(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<32x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x128xf32, #mma1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma1>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %0:3 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %arg3, %arg7 = %arg4, %arg8 = %cst) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma1>) {
      %1 = tt.load %arg6 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %2 = ttg.local_alloc %1 : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared, #smem>
      %3 = tt.load %arg7 {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %4 = ttg.local_alloc %3 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
      %5 = ttng.warp_group_dot %2, %4, %arg8 {maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x32xf16, #shared, #smem> * !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma1>
      %6 = tt.addptr %arg6, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %7 = tt.addptr %arg7, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %6, %7, %5 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma1>
    }
    tt.return %0#2 : tensor<128x128xf32, #mma1>
  }
  tt.func @load_into_lt_4b(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<32x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x128xf32, #mma1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma1>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %0:3 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %arg3, %arg7 = %arg4, %arg8 = %cst) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma1>) {
      %1 = tt.load %arg6 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %2 = ttg.local_alloc %1 : (tensor<128x32xf16, #blocked>) -> !ttg.memdesc<128x32xf16, #shared, #smem>
      %3 = tt.load %arg7 : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %4 = ttg.local_alloc %3 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared1, #smem>
      %5 = ttng.warp_group_dot %2, %4, %arg8 {maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x32xf16, #shared, #smem> * !ttg.memdesc<32x128xf16, #shared1, #smem> -> tensor<128x128xf32, #mma1>
      %6 = tt.addptr %arg6, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %7 = tt.addptr %arg7, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %6, %7, %5 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma1>
    }
    tt.return %0#2 : tensor<128x128xf32, #mma1>
  }
  tt.func @intermediate_use(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<32x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<32x128xf16, #blocked1>
    %0:3 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %arg3, %arg7 = %arg4, %arg8 = %cst) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %1 = tt.load %arg6 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %2 = ttg.convert_layout %1 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %3 = tt.load %arg7 {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %4 = arith.mulf %3, %cst_2 : tensor<32x128xf16, #blocked1>
      %5 = ttg.convert_layout %4 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %6 = tt.dot %2, %5, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %7 = tt.addptr %arg6, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %8 = tt.addptr %arg7, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %7, %8, %6 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    }
    tt.return %0#2 : tensor<128x128xf32, #mma>
  }
  tt.func @indirect_load(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<i32>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<32x128x!tt.ptr<i32>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg6: tensor<32x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %0:5 = scf.for %arg7 = %arg0 to %arg1 step %arg2 iter_args(%arg8 = %arg3, %arg9 = %arg4, %arg10 = %arg5, %arg11 = %arg6, %arg12 = %cst) -> (tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<32x128x!tt.ptr<i32>, #blocked1>, tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %1 = tt.load %arg8 {tt.latency = 1 : i32} : tensor<128x32x!tt.ptr<i32>, #blocked>
      %2 = tt.load %arg9 {tt.latency = 1 : i32} : tensor<32x128x!tt.ptr<i32>, #blocked1>
      %3 = tt.addptr %arg8, %cst_0 : tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<128x32xi32, #blocked>
      %4 = tt.addptr %arg9, %cst_1 : tensor<32x128x!tt.ptr<i32>, #blocked1>, tensor<32x128xi32, #blocked1>
      %5 = tt.addptr %arg10, %1 {tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>} : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %6 = tt.addptr %arg11, %2 {tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>} : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      %7 = tt.load %5 {tt.latency = 1 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %8 = ttg.convert_layout %7 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %9 = tt.load %6 {tt.latency = 1 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %10 = ttg.convert_layout %9 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %11 = tt.dot %8, %10, %arg12 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      scf.yield %3, %4, %5, %6, %11 : tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<32x128x!tt.ptr<i32>, #blocked1>, tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    }
    tt.return %0#4 : tensor<128x128xf32, #mma>
  }
  tt.func @mixed_loads(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<i32>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<32x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %0:4 = scf.for %arg6 = %arg0 to %arg1 step %arg2 iter_args(%arg7 = %arg3, %arg8 = %arg4, %arg9 = %arg5, %arg10 = %cst) -> (tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %1 = tt.load %arg7 {tt.latency = 1 : i32} : tensor<128x32x!tt.ptr<i32>, #blocked>
      %2 = tt.addptr %arg7, %cst_0 : tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<128x32xi32, #blocked>
      %3 = tt.addptr %arg8, %1 {tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>} : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %4 = tt.addptr %arg9, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      %5 = tt.load %3 {tt.latency = 1 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %6 = ttg.convert_layout %5 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %7 = tt.load %4 {tt.latency = 1 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %8 = ttg.convert_layout %7 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %9 = tt.dot %6, %8, %arg10 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      scf.yield %2, %3, %4, %9 : tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    }
    tt.return %0#3 : tensor<128x128xf32, #mma>
  }
  tt.func @per_loop_stages(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<32x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> (tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %0:4 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %arg3, %arg7 = %arg4, %arg8 = %cst, %arg9 = %arg3) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16>, #blocked>) {
      %2 = tt.load %arg6 {tt.latency = 3 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %3 = ttg.convert_layout %2 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %4 = tt.load %arg7 {tt.latency = 3 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %5 = ttg.convert_layout %4 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %6 = tt.dot %3, %5, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %7 = tt.load %arg9 {tt.latency = 3 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      "use"(%7) : (tensor<128x32xf16, #blocked>) -> ()
      %8 = tt.addptr %arg6, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %9 = tt.addptr %arg7, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      %10 = tt.addptr %arg9, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      scf.yield %8, %9, %6, %10 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16>, #blocked>
    } {tt.num_stages = 4 : i32}
    %1:3 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %arg3, %arg7 = %arg4, %arg8 = %cst) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %2 = tt.load %arg6 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %3 = ttg.convert_layout %2 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %4 = tt.load %arg7 {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %5 = ttg.convert_layout %4 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %6 = tt.dot %3, %5, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %7 = tt.addptr %arg6, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %8 = tt.addptr %arg7, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %7, %8, %6 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    }
    tt.return %0#2, %1#2 : tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>
  }
  tt.func @indirect_load_cust_stages(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<i32>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<32x128x!tt.ptr<i32>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg6: tensor<32x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %0:5 = scf.for %arg7 = %arg0 to %arg1 step %arg2 iter_args(%arg8 = %arg3, %arg9 = %arg4, %arg10 = %arg5, %arg11 = %arg6, %arg12 = %cst) -> (tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<32x128x!tt.ptr<i32>, #blocked1>, tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %1 = tt.load %arg8 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<i32>, #blocked>
      %2 = tt.load %arg9 {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<i32>, #blocked1>
      %3 = tt.addptr %arg8, %cst_0 : tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<128x32xi32, #blocked>
      %4 = tt.addptr %arg9, %cst_1 : tensor<32x128x!tt.ptr<i32>, #blocked1>, tensor<32x128xi32, #blocked1>
      %5 = tt.addptr %arg10, %1 {tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>} : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %6 = tt.addptr %arg11, %2 {tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>} : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      %7 = tt.load %5 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %8 = ttg.convert_layout %7 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %9 = tt.load %6 {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %10 = ttg.convert_layout %9 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %11 = tt.dot %8, %10, %arg12 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      scf.yield %3, %4, %5, %6, %11 : tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<32x128x!tt.ptr<i32>, #blocked1>, tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    } {tt.num_stages = 5 : i32}
    tt.return %0#4 : tensor<128x128xf32, #mma>
  }
  tt.func @indirect_load_few_stages(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<i32>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<32x128x!tt.ptr<i32>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg5: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg6: tensor<32x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %0:5 = scf.for %arg7 = %arg0 to %arg1 step %arg2 iter_args(%arg8 = %arg3, %arg9 = %arg4, %arg10 = %arg5, %arg11 = %arg6, %arg12 = %cst) -> (tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<32x128x!tt.ptr<i32>, #blocked1>, tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %1 = tt.load %arg8 : tensor<128x32x!tt.ptr<i32>, #blocked>
      %2 = tt.load %arg9 : tensor<32x128x!tt.ptr<i32>, #blocked1>
      %3 = tt.addptr %arg8, %cst_0 : tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<128x32xi32, #blocked>
      %4 = tt.addptr %arg9, %cst_1 : tensor<32x128x!tt.ptr<i32>, #blocked1>, tensor<32x128xi32, #blocked1>
      %5 = tt.addptr %arg10, %1 {tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>} : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %6 = tt.addptr %arg11, %2 {tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>} : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      %7 = tt.load %5 {tt.latency = 1 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %8 = ttg.convert_layout %7 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %9 = tt.load %6 {tt.latency = 1 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %10 = ttg.convert_layout %9 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %11 = tt.dot %8, %10, %arg12 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      scf.yield %3, %4, %5, %6, %11 : tensor<128x32x!tt.ptr<i32>, #blocked>, tensor<32x128x!tt.ptr<i32>, #blocked1>, tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    } {tt.num_stages = 2 : i32}
    tt.return %0#4 : tensor<128x128xf32, #mma>
  }
  tt.func @non_dot_pipeline(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %0:2 = scf.for %arg4 = %arg0 to %arg1 step %arg2 iter_args(%arg5 = %arg3, %arg6 = %cst) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) {
      %1 = tt.load %arg5 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %2 = ttg.convert_layout %1 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %3 = arith.addf %2, %arg6 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %4 = tt.addptr %arg5, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      scf.yield %4, %3 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    } {tt.num_stages = 3 : i32}
    tt.return %0#1 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  }
  tt.func @no_pipeline(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %0:2 = scf.for %arg4 = %arg0 to %arg1 step %arg2 iter_args(%arg5 = %arg3, %arg6 = %cst) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) {
      %1 = tt.load %arg5 : tensor<128x32x!tt.ptr<f16>, #blocked>
      %2 = ttg.convert_layout %1 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %3 = arith.addf %2, %arg6 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %4 = tt.addptr %arg5, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      scf.yield %4, %3 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    }
    tt.return %0#1 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  }
  tt.func @intermediate_use_cust_stages(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<128x32x!tt.ptr<f16>, #blocked> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}, %arg4: tensor<32x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %cst_2 = arith.constant dense<2.000000e+00> : tensor<32x128xf16, #blocked1>
    %0:3 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %arg3, %arg7 = %arg4, %arg8 = %cst) -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>) {
      %1 = tt.load %arg6 {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %2 = ttg.convert_layout %1 : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %3 = tt.load %arg7 {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %4 = arith.mulf %3, %cst_2 : tensor<32x128xf16, #blocked1>
      %5 = ttg.convert_layout %4 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %6 = tt.dot %2, %5, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %7 = tt.addptr %arg6, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %8 = tt.addptr %arg7, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %7, %8, %6 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>
    } {tt.num_stages = 3 : i32}
    tt.return %0#2 : tensor<128x128xf32, #mma>
  }
}
