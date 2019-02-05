# StateSpaceModels.jl Documentation

## Installation

This package is unregistered so you will need to `Pkg.add` it as follows:
```julia
pkg> add https://github.com/LAMPSPUC/StateSpaceModels.jl.git
```

## Notes

StateSpaceModels.jl is a package for modeling, forecasting and simulating time series in a state space framework based on structural models of Harvey (1989), in which a time series is decomposed in trend, slope and seasonals. Implementations were made based on the book "Time Series Analysis by State Space Methods" (2012) by J. Durbin and S. J. Koopman.

## Features

Current features:
* Basic structural model (level, slope, seasonal)
* Exogenous variables
* Square-root Kalman Filter and smoother
* Big Kappa initialization
* Monte Carlo simulation
* Maximum likelihood estimation
* Multivariate modeling

Future features (work in progress):
* User-defined model
* Cycles
* Forecasting and confidence intervals
* Completion of missing values
* Different models for each variable in multivariate case
* Structural break
* Exact initialization
