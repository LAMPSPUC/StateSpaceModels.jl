# Manual

## Quick Start Guide

Although StateSpaceModels.jl has a lot of functionalities, different models and interfaces 
users usuallly just want to fit a model and analyse the residuals, components and make some forecasts.
The following code is a quick start to perform these tasks

```julia
import Pkg

Pkg.add("StateSpaceModels")

using StateSpaceModels

y = randn(100)

model = LocalLevel(y)

fit!(model)

print_results(model)

forec = forecast(model, 10)

kf = kalman_filter(model)

v = get_innovations(kf)

ks = kalman_smoother(model)

alpha = get_smoothed_state(ks)

using Plots

plot(model, forec)

plotdiagnostics(kf)
```

## Models

The package provides a variaty of pre-defined models. If there is any model that you wish was in the package, feel free to open an issue or pull request.

```@docs
UnobservedComponents
ExponentialSmoothing
SARIMA
DAR
BasicStructural
BasicStructuralExplanatory
LinearRegression
LocalLevel
LocalLevelCycle
LocalLevelExplanatory
LocalLinearTrend
MultivariateBasicStructural
VehicleTracking
```

## Naive models

Naive models are not state space models but are good benchmarks for forecasting, for this reason we implemented them here.

```@docs
Naive
SeasonalNaive
ExperimentalSeasonalNaive
```

## Automatic forecasting

Some models have various parameters and modelling options. The package provides simple functions that
search through different parameters to obtain the best fit for your data without a deeper understanding.
The search procedures can be published in scientific papers or purely heuristic designed by the developers.
In any case if the documentation explains the procedures and indicates if there are any references.

```@docs
auto_ets
auto_arima
```

### Implementing a custom model

Users are able to implement any custom user-defined model.

## Systems

The `StateSpaceModel` matrices are represented as a `StateSpaceSystem`.

```@docs
StateSpaceModels.StateSpaceSystem
LinearUnivariateTimeInvariant
LinearUnivariateTimeVariant
LinearMultivariateTimeInvariant
LinearMultivariateTimeVariant
```

## Hyperparameters

The model hyperparameters are constant (non-time-varying) parameters that are optimized when [`fit!`](@ref) is called. The package provides some useful functions to accelerate experimentation and custom model development.

```@docs
StateSpaceModels.HyperParameters
```

The getters are:
```@docs
get_names
number_hyperparameters
```

The setters are:
```@docs
fix_hyperparameters!
set_initial_hyperparameters!
```

Mappings:
```@docs
constrain_variance!
unconstrain_variance!
constrain_box!
unconstrain_box!
constrain_identity!
unconstrain_identity!
```

## Filters and smoothers

StateSpaceModels.jl lets users define tailor-made filters in an easy manner.

```@docs
UnivariateKalmanFilter
ScalarKalmanFilter
SparseUnivariateKalmanFilter
MultivariateKalmanFilter
FilterOutput
SmootherOutput
kalman_filter
kalman_smoother
get_innovations
get_innovations_variance
get_filtered_state
get_filtered_state_variance
get_predictive_state
get_predictive_state_variance
get_smoothed_state
get_smoothed_state_variance
```

## Fitting and Optimizers

StateSpaceModels.jl has an interface for [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) algorithms. The models can be estimated using different algorithms and tunned to the user needs

```@docs
fit!
Optimizer
print_results
has_fit_methods
isfitted
```

## Forecasting and simulating

StateSpaceModels.jl has functions to make forecasts of the predictive densities multiple steps ahead and to
simulate scenarios based on those forecasts. The package also has a functions to benchmark the model forecasts 
using cross_validation techniques.

```@docs
forecast
simulate_scenarios
cross_validation
```

## Visualization

Some user friendly plot recipes are defined using [RecipesBase.jl](https://github.com/JuliaPlots/RecipesBase.jl). If you have any suggestions do not hesitate to post it as an issue.

```@example
using StateSpaceModels, CSV, DataFrames, Plots

air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(air_passengers.passengers)

model = BasicStructural(log_air_passengers, 12)
fit!(model)
forec = forecast(model, 24)

plot(model, forec; legend = :topleft)
```

```@example
using StateSpaceModels, CSV, DataFrames, Plots

finland_fatalities = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
log_finland_fatalities = log.(finland_fatalities.ff)
model = UnobservedComponents(log_finland_fatalities; trend = "local linear trend")
fit!(model)
ks = kalman_smoother(model)

plot(model, ks)
```

```@example
using StateSpaceModels, CSV, DataFrames, Plots

nile = CSV.File(StateSpaceModels.NILE) |> DataFrame
model = UnobservedComponents(nile.flow; trend = "local level", cycle = "stochastic")
fit!(model)
kf = kalman_filter(model)

plotdiagnostics(kf)
```

## Datasets

The package provides some datasets to illustrate the funtionalities and models. These datasets are stored as csv files and the path to these files can be obtained through their names as seen below. In the examples we illustrate the datasets using [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) and [CSV.jl](https://github.com/JuliaData/CSV.jl)

```@docs
StateSpaceModels.AIR_PASSENGERS
StateSpaceModels.FRONT_REAR_SEAT_KSI
StateSpaceModels.INTERNET
StateSpaceModels.NILE
StateSpaceModels.RJ_TEMPERATURE
StateSpaceModels.VEHICLE_FATALITIES
StateSpaceModels.WHOLESALE_PRICE_INDEX
StateSpaceModels.US_CHANGE
StateSpaceModels.SUNSPOTS_YEAR
```
