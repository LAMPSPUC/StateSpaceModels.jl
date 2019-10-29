module StateSpaceModels

using Optim, Distributions, LinearAlgebra, StaticArrays, Dates, Printf, StatsBase

import Base: size, show, copy

export statespace, kfas, diagnostics

include("prints.jl")
include("structures.jl")
include("utils.jl")
include("model_unknowns.jl")
include("models.jl")
include("estimation.jl")
include("random_seeds_lbfgs.jl")
include("kalman_utils.jl")
include("univariate_kalman.jl")
include("kalman.jl")
include("sqrt_kalman.jl")
include("forecast.jl")
include("diagnostics.jl")

"""
    statespace(model::StateSpaceModel; filter_type::DataType = KalmanFilter, optimization_method::AbstractOptimizationMethod = RandomSeedsLBFGS(), verbose::Int = 1)

Estimate the pre-specified state-space model. The function will only estimate the entries that are declared `NaN`. If there are no NaNs in the `model` it will
only perform the filter and smoother computations.
"""
function statespace(model::StateSpaceModel{T}; filter_type::DataType = KalmanFilter{T}, 
                    optimization_method::AbstractOptimizationMethod = RandomSeedsLBFGS(), verbose::Int = 1) where T

    if !(verbose in [0, 1, 2, 3])
        @warn("Incorrect verbose value input (should be 0, 1, 2 or 3): switching to default value 1")
        verbose = 1
    end

    print_header(verbose)

    # Maximum likelihood estimation
    estimate_statespace!(model, filter_type, optimization_method; verbose = verbose)

    # Kalman filter and smoothing
    filter_output, smoothed_state = kfas(model, filter_type)

    print_bottom(verbose)

    return StateSpace{T}(model, filter_output, smoothed_state, filter_type, optimization_method)
end

"""
    kfas(model::StateSpaceModel{T}, filter_type::DataType) where T

Perform Kalman filter and smoother according to the chosen `filter_type`.
"""
function kfas(model::StateSpaceModel{T}, filter_type::DataType) where T
    error(filter_type , " not implemented") # Returns an error if it cannot
                                            # find a specialized kfas
end

end # end module
