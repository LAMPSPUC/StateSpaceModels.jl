using CSV
using DataFrames
using LinearAlgebra
using StateSpaceModels
using Statistics
using Test

# Functions that are used in different tests
include("utils.jl")

@testset "Models" begin
    include("models/locallevel.jl")
    include("models/locallineartrend.jl")
    include("models/locallevelcycle.jl")
    include("models/basicstructural.jl")
    include("models/basicstructural_multivariate.jl")
    include("models/arima.jl")
    include("models/linear_regression.jl")
end

@testset "Systems" begin
    include("systems.jl")
end
