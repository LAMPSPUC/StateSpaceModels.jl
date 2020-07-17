using Test
using CSV
using DataFrames
using StateSpaceModels

read_csv(file::String) = DataFrame!(CSV.File(file))

@testset "Models" begin
    include("models/locallevel.jl")
    include("models/basicstructural.jl")
    include("models/linear_regression.jl")
end