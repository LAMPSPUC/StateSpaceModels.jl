module StateSpaceModels

abstract type StateSpaceModel end

import Base: show, length

using LinearAlgebra
using Statistics
using Printf
using Optim

include("datasets.jl")

include("hyperparameters.jl")
include("systems.jl")

include("kalman_filter_and_smoother.jl")

include("filters/univariate_kalman_filter.jl")
include("filters/scalar_kalman_filter.jl")
include("filters/regression_kalman_filter.jl")

include("smoothers/kalman_smoother.jl")

include("models/common.jl")
include("models/locallevel.jl")
include("models/locallineartrend.jl")
include("models/damped_lineartrend.jl")
include("models/basicstructural.jl")
include("models/arima.jl")
include("models/linear_regression.jl")

include("optimizers.jl")
include("fit.jl")
include("forecast.jl")

end