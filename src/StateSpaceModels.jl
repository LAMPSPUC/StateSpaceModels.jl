module StateSpaceModels

using Optim, Distributions, LinearAlgebra

export statespace, simulate, structuralmodel

include("structures.jl")
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
    kfilter = sqrt_kalmanfilter(model, param.sqrtH, param.sqrtQ)
    smoothedstate = sqrt_smoother(model, kfilter)

    @info("Filtering and smoothing completed.")

    output = StateSpace(model, smoothedstate, param, kfilter)

    return output
end

end
