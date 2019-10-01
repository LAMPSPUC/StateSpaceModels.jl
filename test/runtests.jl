push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/StateSpaceModels.jl/src")
using Test, StateSpaceModels, Statistics, CSV, LinearAlgebra

const SSM = StateSpaceModels
cd("test")
# Run tests
include("test_utils.jl")
include("test_userdefined.jl")
include("test_locallevel.jl")
include("test_lineartrend.jl")
include("test_structural.jl")
include("test_simulate.jl")
include("test_results.jl")
