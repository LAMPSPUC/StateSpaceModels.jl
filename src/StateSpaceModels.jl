module StateSpaceModels

using Optim, Distributions, LinearAlgebra, StaticArrays

import Base: size

export statespace

include("structures.jl")
include("utils.jl")
include("models.jl")
include("estimation.jl")
include("kalman.jl")
include("simulation.jl")

function statespace(model::StateSpaceModel; nseeds::Int = 3, verbose::Int = 1)

    if !(verbose in [0, 1, 2])
        @warn("Incorrect verbose value input (should be 0, 1, or 2): switching to default value 1")
        verbose = 1
    end

    if verbose > 0
        @info("Starting state-space model estimation...")
    end

    # Maximum likelihood estimation
    param = estimate_statespace(model, nseeds)

    if verbose > 0
        @info("End of estimation.")
        @info("Starting filtering and smoothing...")
    end

    # Kalman filter and smoothing
    kfilter, U2star, K = sqrt_kalmanfilter(model, param.sqrtH, param.sqrtQ)
    smoothedstate = sqrt_smoother(model, kfilter, U2star, K)

    if verbose > 0
        @info("Filtering and smoothing completed.")
    end

    output = StateSpace(model, smoothedstate, param, kfilter)

    return output
end

end
