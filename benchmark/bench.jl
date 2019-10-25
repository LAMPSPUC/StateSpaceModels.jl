using StateSpaceModels, BenchmarkTools, CSV
# Local level series
locallevel_series = CSV.read("./benchmark/locallevel.csv"; header = false)
verbose = 0

locallevel = local_level(Matrix{Float64}(locallevel_series))
locallevel.H[1] = 1
locallevel.Q[1] = 1
@benchmark StateSpaceModels.kalman_filter($locallevel)
# 24th October 2019
# BenchmarkTools.Trial:
#   memory estimate:  12.17 KiB
#   allocs estimate:  93
#   --------------
#   minimum time:     9.258 μs (0.00% GC)
#   median time:      9.832 μs (0.00% GC)
#   mean time:        12.296 μs (11.59% GC)
#   maximum time:     4.382 ms (99.48% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1

locallevel = local_level(Matrix{Float64}(locallevel_series))
locallevel.H[1] = 1
locallevel.Q[1] = 1
@benchmark StateSpaceModels.univariate_kalman_filter($locallevel)
# 24th October 2019
# BenchmarkTools.Trial:
#   memory estimate:  8.33 KiB
#   allocs estimate:  36
#   --------------
#   minimum time:     4.977 μs (0.00% GC)
#   median time:      5.458 μs (0.00% GC)
#   mean time:        6.477 μs (10.23% GC)
#   maximum time:     406.721 μs (97.11% GC)
#   --------------
#   samples:          10000
#   evals/sample:     6

# Benchmark structural model
AP = CSV.read("./examples/AirPassengers.csv")
logAP = log.(Vector{Float64}(AP[:Passengers]))

structural_model = structural(logAP, 12)
structural_model.H[1] = 1e-4
structural_model.Q[[1, 5, 9]] .= 1e-4

@benchmark StateSpaceModels.kalman_filter($structural_model)
# 24th October 2019
# BenchmarkTools.Trial:
#   memory estimate:  740.64 KiB
#   allocs estimate:  1455
#   --------------
#   minimum time:     336.863 μs (0.00% GC)
#   median time:      352.551 μs (0.00% GC)
#   mean time:        473.981 μs (13.59% GC)
#   maximum time:     5.512 ms (77.69% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1

structural_model = structural(logAP, 12)
structural_model.H[1] = 1e-4
structural_model.Q[[1, 5, 9]] .= 1e-4

@benchmark StateSpaceModels.univariate_kalman_filter($structural_model)
# 24th October 2019
# BenchmarkTools.Trial:
#   memory estimate:  658.19 KiB
#   allocs estimate:  446
#   --------------
#   minimum time:     246.795 μs (0.00% GC)
#   median time:      256.044 μs (0.00% GC)
#   mean time:        304.342 μs (12.20% GC)
#   maximum time:     3.262 ms (79.72% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1