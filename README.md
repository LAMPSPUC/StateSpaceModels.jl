# StateSpaceModels.jl

| **DOI** | **Build Status** | **Coverage** | **Documentation** |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1240453.svg)](https://doi.org/10.5281/zenodo.1240453) | [![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url] |[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://lampspuc.github.io/StateSpaceModels.jl/latest/)

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

[build-img]: https://travis-ci.org/LAMPSPUC/StateSpaceModels.jl.svg?branch=master
[build-url]: https://travis-ci.org/LAMPSPUC/StateSpaceModels.jl

[codecov-img]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl?branch=master
