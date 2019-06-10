push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/StateSpaceModels.jl/src")
using StateSpaceModels, BenchmarkTools, CSV
cd("benchmark")
# Local level series
locallevel_series = CSV.read("locallevel.csv"; header = false)
locallevel = locallevelmodel(Matrix{Float64}(locallevel_series))
verbose = 0
bench_local_level = @benchmark statespace($locallevel, verbose = $verbose)
# 10th June 2019
# BenchmarkTools.Trial:
#   memory estimate:  238.07 MiB
#   allocs estimate:  2727909
#   --------------
#   minimum time:     163.971 ms (23.07% GC)
#   median time:      228.808 ms (23.03% GC)
#   mean time:        229.100 ms (23.59% GC)
#   maximum time:     321.260 ms (24.98% GC)
#   --------------
#   samples:          22
#   evals/sample:     1