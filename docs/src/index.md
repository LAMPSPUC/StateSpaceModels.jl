# StateSpaceModels.jl Documentation

## Installation

This package is unregistered so you will need to `Pkg.add` it as follows:
```julia
Pkg.add("https://github.com/LAMPSPUC/StateSpaceModels.jl.git")
```

## Notes

StateSpaceModels.jl is a package for modeling, forecasting and simulating time series in a state space framework. Implementations were made based on the book Time series analysis by state space methods: J. Durbin and S.J. Koopman.

## Features

Current features:
* Basic structural model (level, slope, seasonal)
* Exogenous variables
* Square-root Kalman Filter and smoother
* Big Kappa initialization
* Monte Carlo simulation
* Parallel MLE estimation
* Multivariate modeling

Future features (work in progress):
* User-defined model
* Cycles
* Forecasting and confidence intervals
* Completion of missing values
* Different models for each variable in multivariate case
* Structural break
* Exact initialization