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
# 6th August 2019
# BenchmarkTools.Trial:
#   memory estimate:  106.18 MiB
#   allocs estimate:  1500659
#   --------------
#   minimum time:     80.464 ms (22.75% GC)
#   median time:      110.805 ms (23.33% GC)
#   mean time:        112.070 ms (22.92% GC)
#   maximum time:     131.950 ms (21.28% GC)
#   --------------
#   samples:          45
#   evals/sample:     1

H = fill(1.0, (1,1))
Q = fill(1.0, (1,1))
@benchmark StateSpaceModels.kalman_filter($locallevel, $H, $Q)
# 5th August 2019
# BenchmarkTools.Trial:
#   memory estimate:  91.80 KiB
#   allocs estimate:  1229
#   --------------
#   minimum time:     49.136 μs (0.00% GC)
#   median time:      56.570 μs (0.00% GC)
#   mean time:        73.994 μs (20.43% GC)
#   maximum time:     6.436 ms (97.36% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1