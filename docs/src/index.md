# StateSpaceModels.jl Documentation

## Installation

This package is unregistered so you will need to `Pkg.add` it as follows:
```julia
Pkg.add("https://github.com/LAMPSPUC/StateSpaceModels.jl.git")
```

## Notes

This package is under development and some features may be changed or added.

StateSpaceModels.jl is a package for modeling, forecasting and simulating time series in a state space framework.

Estimation is done through function `statespace` and will automatically be run in parallel with all the currently active threads.

Simulation of scenarios is done through function `simulate`.

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