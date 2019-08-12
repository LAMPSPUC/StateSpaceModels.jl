using StateSpaceModels, BenchmarkTools, CSV
# Local level series
locallevel_series = CSV.read("./benchmark/locallevel.csv"; header = false)
verbose = 0

locallevel = local_level(Matrix{Float64}(locallevel_series))
bench_local_level = @benchmark statespace($locallevel, filter_type = $SquareRootFilter, verbose = $verbose)
# 9th August 2019
# BenchmarkTools.Trial: 
#   memory estimate:  135.70 MiB
#   allocs estimate:  1613460
#   --------------
#   minimum time:     110.164 ms (11.13% GC)
#   median time:      161.256 ms (11.65% GC)
#   mean time:        161.873 ms (11.17% GC)
#   maximum time:     206.068 ms (11.31% GC)
#   --------------
#   samples:          31
#   evals/sample:     1

bench_local_level = @benchmark statespace($locallevel, filter_type = $KalmanFilter, verbose = $verbose)
# 11th August 2019
# BenchmarkTools.Trial: 
#   memory estimate:  10.62 MiB
#   allocs estimate:  92728
#   --------------
#   minimum time:     11.002 ms (0.00% GC)
#   median time:      19.476 ms (7.28% GC)
#   mean time:        20.071 ms (5.96% GC)
#   maximum time:     45.660 ms (3.44% GC)
#   --------------
#   samples:          250
#   evals/sample:     1

H = fill(1.0, (1,1))
Q = fill(1.0, (1,1))
@benchmark StateSpaceModels.kalman_filter($locallevel, $H, $Q)
# 11th August 2019
# BenchmarkTools.Trial: 
#   memory estimate:  12.17 KiB
#   allocs estimate:  93
#   --------------
#   minimum time:     11.164 μs (0.00% GC)
#   median time:      11.530 μs (0.00% GC)
#   mean time:        13.175 μs (5.00% GC)
#   maximum time:     1.549 ms (98.61% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1