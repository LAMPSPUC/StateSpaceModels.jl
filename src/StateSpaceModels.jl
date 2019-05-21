module StateSpaceModels

using Optim, Distributions, LinearAlgebra

export statespace, simulate, StructuralModel

include("structures.jl")
include("models.jl")
include("estimation.jl")
include("kalmanfilter.jl")
include("build.jl")
include("simulation.jl")

function statespace(model::StateSpaceModel; nseeds::Int = 3)

    # Build state-space system
    sys = build_statespace(model)
    
    # Maximum likelihood estimation
    param = estimate_statespace(sys, nseeds)

    # Kalman filter and smoothing
    kfilter = sqrt_kalmanfilter(sys, param.sqrtH, param.sqrtQ)
    smoothedstate = sqrt_smoother(sys, kfilter)

    @info("End of structural model estimation.")

    output = StateSpace(sys, smoothedstate, param, kfilter)

    return output

end

end
