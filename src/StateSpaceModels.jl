module StateSpaceModels

using Optim, Distributions, LinearAlgebra, StaticArrays, Dates, Printf

import Base: size, show

export statespace, kfas

include("prints.jl")
include("structures.jl")
include("utils.jl")
include("models.jl")
include("estimation.jl")
include("random_seeds_lbfgs.jl")
include("kalman_utils.jl")
include("univariate_kalman.jl")
include("kalman.jl")
include("sqrt_kalman.jl")
include("forecast.jl")

"""
    statespace(model::StateSpaceModel; filter_type::DataType = KalmanFilter, optimization_method::AbstractOptimizationMethod = RandomSeedsLBFGS(), verbose::Int = 1)

Estimate the pre-specified state-space model.
"""
function statespace(model::StateSpaceModel; filter_type::DataType = KalmanFilter,
                    optimization_method::AbstractOptimizationMethod = RandomSeedsLBFGS(), verbose::Int = 1)

    if !(verbose in [0, 1, 2])
        @warn("Incorrect verbose value input (should be 0, 1, or 2): switching to default value 1")
        verbose = 1
    end

    print_header(verbose)

    # Maximum likelihood estimation
    covariance = estimate_statespace(model, filter_type, optimization_method; verbose = verbose)

    # Kalman filter and smoothing
    filter_output, smoothed_state = kfas(model, covariance, filter_type)

    print_bottom(verbose)

    return StateSpace(model, filter_output, smoothed_state, covariance, filter_type, optimization_method)
end

"""
    kfas(model::StateSpaceModel, covariance::StateSpaceCovariance,
         filter_type::DataType)

Perform kalman filter and smoother according to the chosen `filter_type`.
"""
function kfas(model::StateSpaceModel, covariance::StateSpaceCovariance,
              filter_type::DataType)
    error(filter_type , " not implemented") # Returns an error if it cannot
                                            # find a specialized kfas
end

end # end module
