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
using OrderedCollections
using RecipesBase
using SeasonalTrendLoess
using SparseArrays
using StatsBase

abstract type StateSpaceModel end

include("datasets.jl")

include("hyperparameters.jl")
include("systems.jl")
include("kalman_filter_and_smoother.jl")
include("statistical_tests/kpss.jl")
include("statistical_tests/canova_hansen.jl")
include("filters/univariate_kalman_filter.jl")
include("filters/multivariate_kalman_filter.jl")
include("filters/scalar_kalman_filter.jl")
include("filters/regression_kalman_filter.jl")
include("filters/sparse_univariate_kalman_filter.jl")

include("smoothers/kalman_smoother.jl")

include("optimizers.jl")
include("fit.jl")
include("prints.jl")
include("forecast.jl")
include("cross_validation.jl")

include("models/common.jl")
include("models/locallevel.jl")
include("models/locallevelcycle.jl")
include("models/locallevelexplanatory.jl")
include("models/locallineartrend.jl")
include("models/basicstructural.jl")
include("models/basicstructural_explanatory.jl")
include("models/basicstructural_multivariate.jl")
include("models/sarima.jl")
include("models/linear_regression.jl")
include("models/unobserved_components.jl")

include("models/exponential_smoothing.jl")
include("models/naive_models.jl")
include("models/dar.jl")
include("models/vehicle_tracking.jl")

include("visualization/forecast.jl")
include("visualization/components.jl")
include("visualization/cross_validation.jl")
include("visualization/diagnostics.jl")

# Exported types and structs
export BasicStructural
export ExperimentalSeasonalNaive
export BasicStructuralExplanatory
export DAR
export ExponentialSmoothing
export FilterOutput
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
export MultivariateKalmanFilter
export Naive
export Optimizer
export SARIMA
export ScalarKalmanFilter
export SeasonalNaive
export SmootherOutput
export SparseUnivariateKalmanFilter
export StateSpaceModel
export UnivariateKalmanFilter
export UnobservedComponents
export VehicleTracking

# Exported functions
export auto_arima
export auto_ets
export cross_validation
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
export get_innovations_variance
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
export num_states
export number_hyperparameters
export print_results
export set_initial_hyperparameters!
export simulate
export simulate_scenarios
export unconstrain_box!
export unconstrain_identity!
export unconstrain_variance!

end
