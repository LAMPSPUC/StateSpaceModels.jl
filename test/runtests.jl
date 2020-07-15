using Test
using DelimitedFiles
using StateSpaceModels

@testset "Models" begin
    include("models/locallevel.jl")
    include("models/basicstructural.jl")
    include("models/linear_regression.jl")
end