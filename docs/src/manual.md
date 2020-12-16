# Manual

## Quick Start Guide

- models
- fit
- filter
- smoother
- forecast
- plot

## Models

The package provides a variaty of pre-defined models. If there is any model that you wish was in the package, feel free to open an issue or pull request.

```@docs
SARIMA
BasicStructural
LinearRegression
LocalLevel
LocalLevelCycle
LocalLevelExplanatory
LocalLinearTrend
MultivariateBasicStructural
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

StateSpaceModels.jl lets user define in an easy manner a tailor made filter. TODO docs here

```@docs
UnivariateKalmanFilter
ScalarKalmanFilter
StateSpaceModels.FilterOutput
get_innovations
get_innovation_variance
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
results
has_fit_methods
is_fitted
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
savefig("plot.png")

nothing
```
![](plot.png)

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
```
