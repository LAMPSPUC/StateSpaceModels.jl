using CSV
using DataFrames
using LinearAlgebra
using RecipesBase
using StateSpaceModels
using Statistics
using Test

# Functions that are used in different tests
include("utils.jl")

# Core functionality
include("systems.jl")

# Models
include("models/locallevel.jl")
include("models/locallineartrend.jl")
include("models/locallevelcycle.jl")
include("models/locallevelexplanatory.jl")
include("models/basicstructural.jl")
include("models/basicstructural_multivariate.jl")
include("models/sarima.jl")
include("models/linear_regression.jl")

# Visualization
include("visualization/forecast.jl")