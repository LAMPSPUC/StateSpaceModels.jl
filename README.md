[build-img]: https://travis-ci.com/LAMPSPUC/StateSpaceModels.jl.svg?branch=master
[build-url]: https://travis-ci.com/LAMPSPUC/StateSpaceModels.jl

[codecov-img]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl?branch=master

# StateSpaceModels.jl

| **Build Status** | **Coverage** | **Documentation** |
|:-----------------:|:-----------------:|:-----------------:|
| [![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url] |[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://lampspuc.github.io/StateSpaceModels.jl/latest/)

StateSpaceModels.jl is a package for modeling, forecasting, and simulating time series in a state-space framework. Implementations were made based on the book "Time Series Analysis by State Space Methods" (2012) by James Durbin and Siem Jan Koopman. The notation of the variables in the code also follows the book.

## Installation

This package is registered in METADATA so you can `Pkg.add` it as follows:
```julia
pkg> add StateSpaceModels
```

## Features

Current features include:
* Kalman filter and smoother
* Maximum likelihood estimation
* Forecasting and Monte Carlo simulation
* User-defined models (user is able to define the system)
* Several predefined models, including:
  1. Basic structural model (trend, slope, seasonal)
  2. Structural model with exogenous variables
  3. Linear trend model
  4. Local level model
* Completion of missing values
* Diagnostics for the residuals of fitted models

## Citing StateSpaceModels.jl

If you use StateSpaceModels.jl in your work, we kindly ask you to cite the following paper ([pdf](https://arxiv.org/pdf/1908.01757.pdf)):

    @article{SaavedraBodinSouto2019,
      title={StateSpaceModels.jl: a Julia Package for Time-Series Analysis in a State-Space Framework},
      author={Raphael Saavedra and Guilherme Bodin and Mario Souto},
      journal={arXiv preprint arXiv:1908.01757},
      year={2019}
    }