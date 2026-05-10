#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32_i32_i32_i32_i32_i32_i32__12c64_13c64_14c64_15c8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32) {
    %cst = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %c64_i32_1 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32_1 : i32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32_1 : i32
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
    %15 = arith.muli %12, %c64_i32_1 : i32
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %17 = tt.splat %15 : i32 -> tensor<64xi32, #blocked1>
    %18 = arith.addi %17, %16 : tensor<64xi32, #blocked1>
    %19 = arith.muli %14, %c64_i32_1 : i32
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %21 = tt.splat %19 : i32 -> tensor<64xi32, #blocked1>
    %22 = arith.addi %21, %20 : tensor<64xi32, #blocked1>
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %24 = ttg.convert_layout %18 : tensor<64xi32, #blocked1> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %25 = tt.expand_dims %24 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %26 = ttg.convert_layout %25 : tensor<64x1xi32, #blocked2> -> tensor<64x1xi32, #blocked3>
    %27 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #blocked3>
    %28 = arith.muli %26, %27 : tensor<64x1xi32, #blocked3>
    %29 = ttg.convert_layout %23 : tensor<64xi32, #blocked1> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi32, #blocked4>
    %31 = ttg.convert_layout %30 : tensor<1x64xi32, #blocked4> -> tensor<1x64xi32, #blocked>
    %32 = tt.splat %arg7 : i32 -> tensor<1x64xi32, #blocked>
    %33 = arith.muli %31, %32 : tensor<1x64xi32, #blocked>
    %34 = tt.broadcast %28 : tensor<64x1xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
    %35 = ttg.convert_layout %34 : tensor<64x64xi32, #blocked3> -> tensor<64x64xi32, #blocked>
    %36 = tt.broadcast %33 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %37 = arith.addi %35, %36 : tensor<64x64xi32, #blocked>
    %38 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %39 = tt.addptr %38, %37 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    %40 = ttg.convert_layout %23 : tensor<64xi32, #blocked1> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %41 = tt.expand_dims %40 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %42 = ttg.convert_layout %41 : tensor<64x1xi32, #blocked2> -> tensor<64x1xi32, #blocked3>
    %43 = tt.splat %arg8 : i32 -> tensor<64x1xi32, #blocked3>
    %44 = arith.muli %42, %43 : tensor<64x1xi32, #blocked3>
    %45 = ttg.convert_layout %22 : tensor<64xi32, #blocked1> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %46 = tt.expand_dims %45 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi32, #blocked4>
    %47 = ttg.convert_layout %46 : tensor<1x64xi32, #blocked4> -> tensor<1x64xi32, #blocked>
    %48 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #blocked>
    %49 = arith.muli %47, %48 : tensor<1x64xi32, #blocked>
    %50 = tt.broadcast %44 : tensor<64x1xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
    %51 = ttg.convert_layout %50 : tensor<64x64xi32, #blocked3> -> tensor<64x64xi32, #blocked>
    %52 = tt.broadcast %49 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %53 = arith.addi %51, %52 : tensor<64x64xi32, #blocked>
    %54 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %55 = tt.addptr %54, %53 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    %56:3 = scf.for %arg12 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg13 = %cst_0, %arg14 = %39, %arg15 = %55) -> (tensor<64x64xf32, #blocked>, tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64x!tt.ptr<f32>, #blocked>)  : i32 {
      %95 = tt.load %arg14, %cst, %cst_0 : tensor<64x64x!tt.ptr<f32>, #blocked>
      %96 = tt.load %arg15, %cst, %cst_0 : tensor<64x64x!tt.ptr<f32>, #blocked>
      %97 = ttg.convert_layout %95 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked5}>>
      %98 = ttg.convert_layout %96 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked5}>>
      %99 = ttg.convert_layout %cst_0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked5>
      %100 = tt.dot %97, %98, %99 : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked5}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked5}>> -> tensor<64x64xf32, #blocked5>
      %101 = ttg.convert_layout %100 : tensor<64x64xf32, #blocked5> -> tensor<64x64xf32, #blocked>
      %102 = arith.addf %arg13, %101 : tensor<64x64xf32, #blocked>
      %103 = arith.muli %arg7, %c64_i32_1 : i32
      %104 = tt.splat %103 : i32 -> tensor<64x64xi32, #blocked>
      %105 = tt.addptr %arg14, %104 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
      %106 = arith.muli %arg8, %c64_i32_1 : i32
      %107 = tt.splat %106 : i32 -> tensor<64x64xi32, #blocked>
      %108 = tt.addptr %arg15, %107 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
      scf.yield %102, %105, %108 : tensor<64x64xf32, #blocked>, tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64x!tt.ptr<f32>, #blocked>
    }
    %57 = arith.muli %12, %c64_i32_1 : i32
    %58 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %59 = tt.splat %57 : i32 -> tensor<64xi32, #blocked1>
    %60 = arith.addi %59, %58 : tensor<64xi32, #blocked1>
    %61 = arith.muli %14, %c64_i32_1 : i32
    %62 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %63 = tt.splat %61 : i32 -> tensor<64xi32, #blocked1>
    %64 = arith.addi %63, %62 : tensor<64xi32, #blocked1>
    %65 = ttg.convert_layout %60 : tensor<64xi32, #blocked1> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %66 = tt.expand_dims %65 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %67 = ttg.convert_layout %66 : tensor<64x1xi32, #blocked2> -> tensor<64x1xi32, #blocked3>
    %68 = tt.splat %arg10 : i32 -> tensor<64x1xi32, #blocked3>
    %69 = arith.muli %68, %67 : tensor<64x1xi32, #blocked3>
    %70 = ttg.convert_layout %64 : tensor<64xi32, #blocked1> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %71 = tt.expand_dims %70 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi32, #blocked4>
    %72 = ttg.convert_layout %71 : tensor<1x64xi32, #blocked4> -> tensor<1x64xi32, #blocked>
    %73 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #blocked>
    %74 = arith.muli %72, %73 : tensor<1x64xi32, #blocked>
    %75 = tt.broadcast %69 : tensor<64x1xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
    %76 = ttg.convert_layout %75 : tensor<64x64xi32, #blocked3> -> tensor<64x64xi32, #blocked>
    %77 = tt.broadcast %74 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %78 = arith.addi %76, %77 : tensor<64x64xi32, #blocked>
    %79 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %80 = tt.addptr %79, %78 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    %81 = ttg.convert_layout %60 : tensor<64xi32, #blocked1> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %82 = tt.expand_dims %81 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %83 = ttg.convert_layout %82 : tensor<64x1xi32, #blocked2> -> tensor<64x1xi32, #blocked3>
    %84 = tt.splat %arg3 : i32 -> tensor<64x1xi32, #blocked3>
    %85 = arith.cmpi slt, %83, %84 : tensor<64x1xi32, #blocked3>
    %86 = ttg.convert_layout %64 : tensor<64xi32, #blocked1> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %87 = tt.expand_dims %86 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi32, #blocked4>
    %88 = ttg.convert_layout %87 : tensor<1x64xi32, #blocked4> -> tensor<1x64xi32, #blocked>
    %89 = tt.splat %arg4 : i32 -> tensor<1x64xi32, #blocked>
    %90 = arith.cmpi slt, %88, %89 : tensor<1x64xi32, #blocked>
    %91 = tt.broadcast %85 : tensor<64x1xi1, #blocked3> -> tensor<64x64xi1, #blocked3>
    %92 = ttg.convert_layout %91 : tensor<64x64xi1, #blocked3> -> tensor<64x64xi1, #blocked>
    %93 = tt.broadcast %90 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
    %94 = arith.andi %92, %93 : tensor<64x64xi1, #blocked>
    tt.store %80, %56#0, %94 : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

