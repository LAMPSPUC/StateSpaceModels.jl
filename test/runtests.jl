using CSV
using DataFrames
using Documenter
using LinearAlgebra
using StateSpaceModels
using Statistics
using Test

# Set up to run docstrings with jldoctest
DocMeta.setdocmeta!(StateSpaceModels, :DocTestSetup, :(using StateSpaceModels); recursive=true)

# Functions that are used in different tests
include("utils.jl")

@testset "Models" begin
    include("models/locallevel.jl")
    include("models/basicstructural.jl")
    include("models/arima.jl")
    include("models/linear_regression.jl")
end

@testset "Systems" begin
    include("systems.jl")
end

@testset "Documentation examples" begin
    doctest(StateSpaceModels)
end