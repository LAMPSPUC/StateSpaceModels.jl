
path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)

using Test, StateSpaceModels, Statistics, CSV

# Run tests
# include("test_userdefined.jl")
# include("test_locallevel.jl")
# include("test_lineartrend.jl")
include("test_statespace.jl")
# include("test_utils.jl")
