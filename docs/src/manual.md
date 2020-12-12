# Manual

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

The model hyperparameters are constant (non-time-varying) parameters that are optimized when `fit!` is called.
The package provides some useful getters and setters to accelerate experimentation with models.

The getters are:
```@docs
get_hyperparameters
get_names
```

The setters are:
```@docs
fix_hyperparameters!
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

## Optim.jl interface

The optimizer to be used can be configured through an interface with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl):

```@docs
Optimizer
```

## Datasets

The package provides some datasets to illustrate the funtionalities and models. 
These datasets are stored as csv files and the path to these files can be obtained through their names as seen below.
In the examples we illustrate the datasets using [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) and [CSV.jl](https://github.com/JuliaData/CSV.jl)

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
