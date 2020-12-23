module StateSpaceModels

import Base: show, length, isempty

using Distributions
using LinearAlgebra
using ShiftedArrays
using Statistics
using Polynomials
using MatrixEquations
using Printf
using Optim
using RecipesBase

abstract type StateSpaceModel end

include("datasets.jl")

include("hyperparameters.jl")
include("systems.jl")
include("kalman_filter_and_smoother.jl")

include("filters/univariate_kalman_filter.jl")
include("filters/multivariate_kalman_filter.jl")
include("filters/scalar_kalman_filter.jl")
include("filters/regression_kalman_filter.jl")

include("smoothers/kalman_smoother.jl")

include("optimizers.jl")
include("fit.jl")
include("prints.jl")
include("forecast.jl")

include("models/common.jl")
include("models/locallevel.jl")
include("models/locallevelcycle.jl")
include("models/locallevelexplanatory.jl")
include("models/locallineartrend.jl")
include("models/basicstructural.jl")
include("models/basicstructural_multivariate.jl")
include("models/sarima.jl")
include("models/linear_regression.jl")
include("models/unobserved_components.jl")

include("visualization/forecast.jl")

# Exported types and structs
export SARIMA
export BasicStructural
export LinearMultivariateTimeInvariant
export LinearMultivariateTimeVariant
export LinearRegression
export LinearUnivariateTimeInvariant
export LinearUnivariateTimeVariant
export LocalLevel
export LocalLevelCycle
export LocalLevelExplanatory
export LocalLinearTrend
export MultivariateBasicStructural
export Optimizer
export ScalarKalmanFilter
export StateSpaceModel
export UnivariateKalmanFilter
export UnobservedComponents

# Exported functions
export constrain_box!
export constrain_identity!
export constrain_variance!
export fit!
export fix_hyperparameters!
export forecast
export forecast_expected_value
export get_constrained_value
export get_filtered_state
export get_filtered_state_variance
export get_hyperparameters
export get_innovations
export get_innovation_variance
export get_names
export get_predictive_state
export get_predictive_state_variance
export get_smoothed_state
export get_smoothed_state_variance
export has_fit_methods
export isfitted
export isunivariate
export kalman_filter
export kalman_smoother
export loglike
export number_hyperparameters
export results
export set_initial_hyperparameters!
export simulate
export simulate_scenarios
export unconstrain_box!
export unconstrain_identity!
export unconstrain_variance!

end
