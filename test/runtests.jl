using CSV
using DataFrames
using LinearAlgebra
using RecipesBase
using Statistics
using Test
using Dates
include("/Users/guilhermebodin/Documents/StateSpaceModels.jl/src/StateSpaceModels.jl")

# Functions that are used in different tests
include("test/utils.jl")
include("test/models/unobserved_components.jl")
# Core functionality
# include("test/systems.jl")
# include("test/prints.jl")

# Models
# include("test/models/locallevel.jl")
# include("models/locallineartrend.jl")
# include("models/locallevelcycle.jl")
# include("models/locallevelexplanatory.jl")
# include("models/basicstructural.jl")
# include("models/basicstructural_multivariate.jl")
# include("models/sarima.jl")
include("models/unobserved_components.jl")
# include("models/linear_regression.jl")

# Visualization
include("visualization/forecast.jl")
include("visualization/unobserved_components.jl")