module StateSpaceModels

using Optim, Distributions, LinearAlgebra

import Base: size

export statespace

include("structures.jl")
include("utils.jl")
include("models.jl")
include("estimation.jl")
include("kalman.jl")
include("simulation.jl")

function statespace(model::StateSpaceModel; nseeds::Int = 3)

    @info("Starting state-space model estimation...")

    # Maximum likelihood estimation
    param = estimate_statespace(model, nseeds)

    @info("End of estimation.")
    @info("Starting filtering and smoothing...")

    # Kalman filter and smoothing
    kfilter, U2star, K = sqrt_kalmanfilter(model, param.sqrtH, param.sqrtQ)
    smoothedstate = sqrt_smoother(model, kfilter, U2star, K)

    @info("Filtering and smoothing completed.")

    output = StateSpace(model, smoothedstate, param, kfilter)

    return output
end

end
