module StateSpaceModels

using Optim, Distributions, LinearAlgebra, StaticArrays

using TimerOutputs

import Base: size

export statespace

include("structures.jl")
include("utils.jl")
include("models.jl")
include("estimation.jl")
include("random_seeds_lbfgs.jl")
include("kalman.jl")
include("sqrt_kalman.jl")
include("simulation.jl")

function statespace(model::StateSpaceModel; verbose::Int = 1)

    reset_timer!()

    if !(verbose in [0, 1, 2])
        @warn("Incorrect verbose value input (should be 0, 1, or 2): switching to default value 1")
        verbose = 1
    end

    if verbose > 0
        @info("Starting state-space model estimation.")
    end

    # Maximum likelihood estimation
    covariance = estimate_statespace(model, model.optimization_method; verbose = verbose)

    if verbose > 0
        @info("End of estimation.")
        @info("Starting filtering and smoothing.")
    end

    # Kalman filter and smoothing
    filtered_state, smoothed_state = kalman_filter_and_smoother(model, covariance, model.filter_type)

    return StateSpace(model, filtered_state, smoothed_state, covariance)
end

function kalman_filter_and_smoother(model::StateSpaceModel, covariance::StateSpaceCovariance, 
                                    filter_type::DataType)
    error(filter_type , " not implemented") # Returns an error if it cannot 
                                            # find a specialized kalman_filter_and_smoother
end

end # end module
