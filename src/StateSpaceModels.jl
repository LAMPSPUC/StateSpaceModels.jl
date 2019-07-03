module StateSpaceModels

using Optim, Distributions, LinearAlgebra, StaticArrays, Dates, Printf

import Base: size, show

export statespace

include("prints.jl")
include("structures.jl")
include("utils.jl")
include("models.jl")
include("estimation.jl")
include("random_seeds_lbfgs.jl")
include("kalman.jl")
include("sqrt_kalman.jl")
include("forecast.jl")

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
    filtered_state, smoothed_state = kalman_filter_and_smoother(model, covariance, filter_type)

    print_bottom(verbose)
    return StateSpace(model, filtered_state, smoothed_state, covariance, filter_type, optimization_method)
end

function kalman_filter_and_smoother(model::StateSpaceModel, covariance::StateSpaceCovariance, 
                                    filter_type::DataType)
    error(filter_type , " not implemented") # Returns an error if it cannot 
                                            # find a specialized kalman_filter_and_smoother
end

end # end module
