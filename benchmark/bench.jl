using StateSpaceModels, BenchmarkTools, CSV
# Local level series
locallevel_series = CSV.read("locallevel.csv"; header = false)
verbose = 0

locallevel = local_level(Matrix{Float64}(locallevel_series), filter_type = StateSpaceModels.SquareRootFilter)
bench_local_level = @benchmark statespace($locallevel, verbose = $verbose)
# 28th June 2019
# BenchmarkTools.Trial:
#   memory estimate:  200.64 MiB
#   allocs estimate:  2356684
#   --------------
#   minimum time:     138.056 ms (16.78% GC)
#   median time:      195.104 ms (18.21% GC)
#   mean time:        202.507 ms (17.56% GC)
#   maximum time:     310.801 ms (17.93% GC)
#   --------------
#   samples:          25
#   evals/sample:     1

locallevel = local_level(Matrix{Float64}(locallevel_series), filter_type = StateSpaceModels.KalmanFilter)
bench_local_level = @benchmark statespace($locallevel, verbose = $verbose)
# 28th June 2019
# BenchmarkTools.Trial:
#   memory estimate:  165.57 MiB
#   allocs estimate:  1834110
#   --------------
#   minimum time:     105.676 ms (18.00% GC)
#   median time:      150.408 ms (16.63% GC)
#   mean time:        152.375 ms (18.03% GC)
#   maximum time:     230.527 ms (45.56% GC)
#   --------------
#   samples:          33
#   evals/sample:     1