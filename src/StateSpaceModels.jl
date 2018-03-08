module StateSpaceModels

using Optim, Distributions

export statespace, simulate

include("structures.jl")
include("estimation.jl")
include("kalmanfilter.jl")
include("modeling.jl")
include("simulation.jl")

end
