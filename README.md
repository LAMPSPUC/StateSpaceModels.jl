# StateSpaceModels.jl

| **DOI** | **Build Status** | **Coverage** |
|:-----------------:|:-----------------:|:-----------------:|
| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1240453.svg)](https://doi.org/10.5281/zenodo.1240453) | [![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url] |

## Installation
This package is unregistered so you will need to `Pkg.clone` it as follows:
```julia
Pkg.clone("https://github.com/LAMPSPUC/StateSpaceModels.jl.git")
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

[build-img]: https://travis-ci.org/LAMPSPUC/StateSpaceModels.jl.svg?branch=master
[build-url]: https://travis-ci.org/LAMPSPUC/StateSpaceModels.jl

[codecov-img]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl?branch=master
