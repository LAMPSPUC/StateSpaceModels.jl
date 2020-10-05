# Manual

## Models

The package provides a variaty of pre-defined models, fell free to contribute if you want to add one to the list.

```@docs
ARIMA
BasicStructural
LinearRegression
LocalLevel
LocalLevelCycle
LocalLinearTrend
MultivariateBasicStructural
```

### Implementing a custom StateSpaceModel

## Systems

You can represent the `StateSpaceModel` matrices as a `StateSpaceSystem`. 

```@docs
StateSpaceModels.StateSpaceSystem
LinearUnivariateTimeInvariant
```

## Hyperparameters

StateSpaceModels hyperparameters are variables that are optimized when `fit!` is called.
The package defines some useful getters and setters to accelerate experimentation with 
models.

The getters are
```@docs
get_names
```

The setters are
```@docs
fix_hyperparameters!
```

mappings
```@docs
constrain_variance!
unconstrain_variance!
constrain_box!
unconstrain_box!
constrain_identity!
unconstrain_identity!
```
## Optim interface

```@docs
Optimizer
```

## Datasets

The package provides some datasets to illustrate the funtionalities and models. The datasets 
are strings with the absolute path of the file inside your computer, they are all stored as 
csv files and you may read them the way that fits you better. In the examples we illustrate the
datasets using [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) and [CSV.jl](https://github.com/JuliaData/CSV.jl)

```@docs
StateSpaceModels.NILE
StateSpaceModels.AIR_PASSENGERS
StateSpaceModels.INTERNET
StateSpaceModels.WHOLESALE_PRICE_INDEX
StateSpaceModels.VEHICLE_FATALITIES
```