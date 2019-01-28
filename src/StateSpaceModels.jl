__precompile__()

module StateSpaceModels

using Optim, Distributions, Distributed, LinearAlgebra, SharedArrays

export statespace, simulate

include("structures.jl")
include("estimation.jl")
include("kalmanfilter.jl")
include("modeling.jl")
include("simulation.jl")

end
