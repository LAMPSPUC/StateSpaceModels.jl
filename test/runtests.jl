using Test
using CSV
using DataFrames
using LinearAlgebra
using StateSpaceModels

include("utils.jl")

@testset "Models" begin
    include("models/locallevel.jl")
    include("models/basicstructural.jl")
    include("models/linear_regression.jl")
end