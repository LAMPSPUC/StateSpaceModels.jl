push!(LOAD_PATH, "/Users/guilhermebodin/Documents/StateSpaceModels.jl/src")
using Test, StateSpaceModels, Statistics, CSV, LinearAlgebra
cd("test")
const SSM = StateSpaceModels
# Run tests
include("test_utils.jl")
include("test_userdefined.jl")
include("test_locallevel.jl")
include("test_lineartrend.jl")
include("test_structural.jl")
include("test_simulate.jl")
include("test_results.jl")
