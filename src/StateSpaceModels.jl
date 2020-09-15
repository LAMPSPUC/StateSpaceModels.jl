module StateSpaceModels

import Base: show, length, isempty

using Distributions
using LinearAlgebra
using Statistics
using Printf
using Optim

abstract type StateSpaceModel end

include("datasets.jl")

include("hyperparameters.jl")
include("systems.jl")
include("kalman_filter_and_smoother.jl")

include("filters/univariate_kalman_filter.jl")
include("filters/scalar_kalman_filter.jl")
include("filters/regression_kalman_filter.jl")

include("smoothers/kalman_smoother.jl")

include("models/common.jl")
include("models/locallevel.jl")
include("models/locallineartrend.jl")
include("models/damped_lineartrend.jl")
include("models/basicstructural.jl")
include("models/arima.jl")
include("models/linear_regression.jl")

include("prints.jl")
include("optimizers.jl")
include("fit.jl")
include("forecast.jl")

export ARIMA
export BasicStructural
export covariance_filtered_estimates
export covariance_one_step_ahead_predictions
export covariance_smoothed_estimates
export filtered_estimates
export fit!
export forecast
export forecast_expected_value
export get_constrained_value
export get_hyperparameters
export get_minimizer_hyperparameter_position
export has_fit_methods
export fix_hyperparameters!
export kalman_filter
export kalman_smoother
export LinearMultivariateTimeInvariant
export LinearMultivariateTimeVariant
export LinearRegression
export LinearUnivariateTimeInvariant
export LinearUnivariateTimeVariant
export LocalLevel
export LocalLinearTrend
export loglike
export one_step_ahead_predictions
export Optimizer
export results
export ScalarKalmanFilter
export set_initial_hyperparameters!
export simulate
export simulate_scenarios
export smoothed_estimates

end
