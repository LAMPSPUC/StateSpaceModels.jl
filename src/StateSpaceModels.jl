module StateSpaceModels

abstract type StateSpaceModel end

using LinearAlgebra
using Statistics
using Optim

include("hyperparameters.jl")
include("systems.jl")

include("kalman_filter_and_smoother.jl")

include("filters/univariate_kalman_filter.jl")
include("filters/scalar_kalman_filter.jl")
include("filters/regression_kalman_filter.jl")

include("models/common.jl")
include("models/locallevel.jl")
include("models/lineartrend.jl")
include("models/damped_lineartrend.jl")
include("models/basicstructural.jl")
include("models/arima.jl")
include("models/linear_regression.jl")

include("optimizers.jl")
include("fit.jl")

end