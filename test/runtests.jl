using CSV
using DataFrames
using LinearAlgebra
using Random
using RecipesBase
using StateSpaceModels
using Statistics
using Test

# Functions that are used in different tests
include("utils.jl")

# Core functionality
include("systems.jl")
include("prints.jl")

# Models
include("models/locallevel.jl")
include("models/locallineartrend.jl")
include("models/locallevelcycle.jl")
include("models/locallevelexplanatory.jl")
include("models/basicstructural.jl")
include("models/basicstructural_explanatory.jl")
include("models/basicstructural_multivariate.jl")
include("models/sarima.jl")
include("models/unobserved_components.jl")
include("models/linear_regression.jl")
include("models/exponential_smoothing.jl")
include("models/naive_models.jl")
include("models/dar.jl")
include("models/vehicle_tracking.jl")

# Visualization
include("visualization/forecast.jl")
include("visualization/components.jl")
include("visualization/cross_validation.jl")
include("visualization/diagnostics.jl")