#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32_i32_i32_i32_i32_i32_i32__12c64_13c64_14c64_15c8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32) {
    %c8_i32 = arith.constant 8 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.cmpi slt, %8, %c8_i32 : i32
    %10 = arith.select %9, %8, %c8_i32 : i32
    %11 = arith.remsi %0, %10 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.remsi %0, %5 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c64_i32 : i32
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.splat %15 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %18 = arith.addi %17, %16 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.muli %14, %c64_i32 : i32
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %21 = tt.splat %19 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %22 = arith.addi %21, %20 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %23 = tt.expand_dims %18 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %24 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #blocked1>
    %25 = arith.muli %23, %24 : tensor<64x1xi32, #blocked1>
    %26 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %27 = tt.expand_dims %26 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %28 = tt.splat %arg7 : i32 -> tensor<1x64xi32, #blocked1>
    %29 = arith.muli %27, %28 : tensor<1x64xi32, #blocked1>
    %30 = tt.broadcast %25 : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %31 = tt.broadcast %29 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %32 = arith.addi %30, %31 : tensor<64x64xi32, #blocked1>
    %33 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
    %34 = tt.addptr %33, %32 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
    %35 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %36 = tt.expand_dims %35 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %37 = tt.splat %arg8 : i32 -> tensor<64x1xi32, #blocked1>
    %38 = arith.muli %36, %37 : tensor<64x1xi32, #blocked1>
    %39 = tt.expand_dims %22 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %40 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #blocked1>
    %41 = arith.muli %39, %40 : tensor<1x64xi32, #blocked1>
    %42 = tt.broadcast %38 : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %43 = tt.broadcast %41 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %44 = arith.addi %42, %43 : tensor<64x64xi32, #blocked1>
    %45 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
    %46 = tt.addptr %45, %44 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
    %47:3 = scf.for %arg12 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg13 = %cst, %arg14 = %34, %arg15 = %46) -> (tensor<64x64xf32, #blocked>, tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64x!tt.ptr<f32>, #blocked1>)  : i32 {
      %79 = tt.load %arg14 : tensor<64x64x!tt.ptr<f32>, #blocked1>
      %80 = tt.load %arg15 : tensor<64x64x!tt.ptr<f32>, #blocked1>
      %81 = ttg.convert_layout %79 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %82 = ttg.convert_layout %80 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %83 = tt.dot %81, %82, %cst : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %84 = arith.addf %arg13, %83 : tensor<64x64xf32, #blocked>
      %85 = arith.muli %arg7, %c64_i32 : i32
      %86 = tt.splat %85 : i32 -> tensor<64x64xi32, #blocked1>
      %87 = tt.addptr %arg14, %86 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
      %88 = arith.muli %arg8, %c64_i32 : i32
      %89 = tt.splat %88 : i32 -> tensor<64x64xi32, #blocked1>
      %90 = tt.addptr %arg15, %89 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
      scf.yield %84, %87, %90 : tensor<64x64xf32, #blocked>, tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64x!tt.ptr<f32>, #blocked1>
    }
    %48 = ttg.convert_layout %47#0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    %49 = arith.muli %12, %c64_i32 : i32
    %50 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %51 = tt.splat %49 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %52 = arith.addi %51, %50 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %53 = arith.muli %14, %c64_i32 : i32
    %54 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %55 = tt.splat %53 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %56 = arith.addi %55, %54 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %57 = tt.expand_dims %52 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %58 = tt.splat %arg10 : i32 -> tensor<64x1xi32, #blocked1>
    %59 = arith.muli %58, %57 : tensor<64x1xi32, #blocked1>
    %60 = tt.expand_dims %56 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %61 = ttg.convert_layout %60 : tensor<1x64xi32, #blocked2> -> tensor<1x64xi32, #blocked1>
    %62 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #blocked1>
    %63 = arith.muli %61, %62 : tensor<1x64xi32, #blocked1>
    %64 = tt.broadcast %59 : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %65 = tt.broadcast %63 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %66 = arith.addi %64, %65 : tensor<64x64xi32, #blocked1>
    %67 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
    %68 = tt.addptr %67, %66 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
    %69 = tt.expand_dims %52 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %70 = tt.splat %arg3 : i32 -> tensor<64x1xi32, #blocked1>
    %71 = arith.cmpi slt, %69, %70 : tensor<64x1xi32, #blocked1>
    %72 = tt.expand_dims %56 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %73 = ttg.convert_layout %72 : tensor<1x64xi32, #blocked2> -> tensor<1x64xi32, #blocked1>
    %74 = tt.splat %arg4 : i32 -> tensor<1x64xi32, #blocked1>
    %75 = arith.cmpi slt, %73, %74 : tensor<1x64xi32, #blocked1>
    %76 = tt.broadcast %71 : tensor<64x1xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
    %77 = tt.broadcast %75 : tensor<1x64xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
    %78 = arith.andi %76, %77 : tensor<64x64xi1, #blocked1>
    tt.store %68, %48, %78 : tensor<64x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

