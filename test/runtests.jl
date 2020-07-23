using Test
using CSV
using DataFrames
using LinearAlgebra
using Documenter
using StateSpaceModels

# Set up to run docstrings with jldoctest
DocMeta.setdocmeta!(StateSpaceModels, :DocTestSetup, :(using StateSpaceModels); recursive=true)

# Functions that are used in different tests
include("utils.jl")

@testset "Models" begin
    include("models/locallevel.jl")
    include("models/basicstructural.jl")
    include("models/linear_regression.jl")
end

@testset "Documentation examples" begin
    doctest(StateSpaceModels)
end