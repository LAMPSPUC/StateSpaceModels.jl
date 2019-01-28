push!(LOAD_PATH, "/Users/guilhermebodin/Documents/StateSpaceModels.jl/src/")
using Test, StateSpaceModels, Statistics
cd("./test/")
# Run tests
include("test_statespace.jl")
