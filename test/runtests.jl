
path = joinpath(dirname(@__FILE__), "..", "..")
push!(Base.LOAD_PATH, path)

using Test, StateSpaceModels, Statistics, CSV

using TimerOutputs

# Run tests

reset_timer!()
@timeit "section1" include("test_userdefined.jl")
@timeit "section2" include("test_locallevel.jl")
@timeit "section3" include("test_lineartrend.jl")
@timeit "section4" include("test_statespace.jl")
@timeit "section5" include("test_utils.jl")

print_timer()