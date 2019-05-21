module StateSpaceModels

using Optim, Distributions, LinearAlgebra

export statespace, simulate

include("structures.jl")
include("estimation.jl")
include("kalmanfilter.jl")
include("modeling.jl")
include("simulation.jl")

function statespace(sys::StateSpaceSystem)
    
    # Maximum likelihood estimation
    ss_par = estimate_statespace(sys, nseeds)

    # Kalman filter and smoothing
    ss_filter = sqrt_kalmanfilter(sys, ss_par.sqrtH, ss_par.sqrtQ)
    smoothedstate = sqrt_smoother(sys, ss_filter)

    @info("End of structural model estimation.")

    output = StateSpace(sys, smoothedstate, ss_par, ss_filter)

    return output

end

end
