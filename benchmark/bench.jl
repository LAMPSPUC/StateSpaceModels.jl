using StateSpaceModels, BenchmarkTools, CSV
# Local level series
locallevel_series = CSV.read("./benchmark/locallevel.csv"; header = false)
verbose = 0

locallevel = local_level(Matrix{Float64}(locallevel_series))
bench_local_level = @benchmark statespace($locallevel, filter_type = $SquareRootFilter, verbose = $verbose)
# 5th August 2019
# BenchmarkTools.Trial:
#   memory estimate:  187.38 MiB
#   allocs estimate:  2398549
#   --------------
#   minimum time:     127.855 ms (21.46% GC)
#   median time:      169.408 ms (20.37% GC)
#   mean time:        167.529 ms (20.88% GC)
#   maximum time:     213.863 ms (21.28% GC)
#   --------------
#   samples:          30
#   evals/sample:     1

bench_local_level = @benchmark statespace($locallevel, filter_type = $KalmanFilter, verbose = $verbose)
# 5th August 2019
# BenchmarkTools.Trial:
#   memory estimate:  128.48 MiB
#   allocs estimate:  1832786
#   --------------
#   minimum time:     88.344 ms (25.22% GC)
#   median time:      132.428 ms (24.98% GC)
#   mean time:        133.527 ms (24.29% GC)
#   maximum time:     174.321 ms (23.58% GC)
#   --------------
#   samples:          38
#   evals/sample:     1

H = fill(1.0, (1,1))
Q = fill(1.0, (1,1))
@benchmark StateSpaceModels.kalman_filter($locallevel, $H, $Q)
# 5th August 2019
# BenchmarkTools.Trial:
#   memory estimate:  128.42 KiB
#   allocs estimate:  1761
#   --------------
#   minimum time:     63.112 μs (0.00% GC)
#   median time:      69.907 μs (0.00% GC)
#   mean time:        93.078 μs (21.86% GC)
#   maximum time:     7.084 ms (98.82% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1