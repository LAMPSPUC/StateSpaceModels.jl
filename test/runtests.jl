push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/StateSpaceModels.jl/src")
using Test, StateSpaceModels, Statistics, CSV
cd("test")
# Run tests
include("test_userdefined.jl")
include("test_locallevel.jl")
include("test_lineartrend.jl")
include("test_statespace.jl")
include("test_utils.jl")
