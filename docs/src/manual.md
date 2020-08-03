# Manual

## Models

The package provides a variaty of pre-defined models, fell free to contribute if you want to add one to the list.

### Local Level
```@docs
LocalLevel
```

### BasicStructural
```@docs
BasicStructural
```

### Implementing a custom StateSpaceModel

## Systems

You can represent the `StateSpaceModel` matrices as a `StateSpaceSystem`. 

```@docs
StateSpaceModels.StateSpaceSystem
LinearUnivariateTimeInvariant
```

## Hyperparameters

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
```