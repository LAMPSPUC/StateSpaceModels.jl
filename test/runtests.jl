using Test, StateSpaceModels, Statistics, CSV, LinearAlgebra, Random, Distributions

const SSM = StateSpaceModels

# Run tests
include("test_utils.jl")
include("test_userdefined.jl")
include("test_locallevel.jl")
include("test_lineartrend.jl")
include("test_structural.jl")
include("test_regression.jl")
include("test_simulate.jl")
include("test_opt_methods.jl")
include("test_results.jl")
