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
# 9th August 2019
# BenchmarkTools.Trial: 
#   memory estimate:  44.20 MiB
#   allocs estimate:  564867
#   --------------
#   minimum time:     34.680 ms (11.09% GC)
#   median time:      53.984 ms (10.75% GC)
#   mean time:        53.772 ms (10.87% GC)
#   maximum time:     69.858 ms (11.15% GC)
#   --------------
#   samples:          93
#   evals/sample:     1

H = fill(1.0, (1,1))
Q = fill(1.0, (1,1))
@benchmark StateSpaceModels.kalman_filter($locallevel, $H, $Q)
# 9th August 2019
# BenchmarkTools.Trial: 
#   memory estimate:  66.80 KiB
#   allocs estimate:  829
#   --------------
#   minimum time:     41.687 μs (0.00% GC)
#   median time:      42.620 μs (0.00% GC)
#   mean time:        51.369 μs (11.04% GC)
#   maximum time:     3.085 ms (97.27% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1