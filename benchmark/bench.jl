using StateSpaceModels, BenchmarkTools, CSV
# Local level series
locallevel_series = CSV.read("./benchmark/locallevel.csv"; header = false)
verbose = 0

locallevel = local_level(Matrix{Float64}(locallevel_series))
bench_local_level = @benchmark statespace($locallevel, filter_type = $SquareRootFilter, verbose = $verbose)
# 1st August 2019
# BenchmarkTools.Trial:
#   memory estimate:  555.70 MiB
#   allocs estimate:  2964817
#   --------------
#   minimum time:     248.796 ms (23.64% GC)
#   median time:      306.473 ms (23.65% GC)
#   mean time:        330.366 ms (24.22% GC)
#   maximum time:     487.338 ms (28.60% GC)
#   --------------
#   samples:          16
#   evals/sample:     1

bench_local_level = @benchmark statespace($locallevel, filter_type = $KalmanFilter, verbose = $verbose)
# 1st August 2019
# BenchmarkTools.Trial:
#   memory estimate:  135.35 MiB
#   allocs estimate:  1956723
#   --------------
#   minimum time:     91.660 ms (24.23% GC)
#   median time:      130.084 ms (25.73% GC)
#   mean time:        128.891 ms (24.95% GC)
#   maximum time:     172.001 ms (24.89% GC)
#   --------------
#   samples:          39
#   evals/sample:     1