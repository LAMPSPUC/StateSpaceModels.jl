# StateSpaceModels.jl

| **DOI** | **Build Status** | **Coverage** | **Documentation** |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2598488.svg)](https://doi.org/10.5281/zenodo.2598488) | [![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url] |[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://lampspuc.github.io/StateSpaceModels.jl/latest/)

## Installation

This package is registered in METADATA so you can `Pkg.add` it as follows:
```julia
pkg> add StateSpaceModels
```

## Notes

StateSpaceModels.jl is a package for modeling, forecasting and simulating time series in a state-space framework. Implementations were made based on the book "Time Series Analysis by State Space Methods" (2012) by J. Durbin and S. J. Koopman. The notation of the variables in the code also follows the book.

Works using this package:

[Simulating Low and High-Frequency Energy
Demand Scenarios in a Unified Framework – Part
I: Low-Frequency Simulation](https://proceedings.science/sbpo/papers/simulando-cenarios-de-demanda-em-baixa-e-alta-frequencia-em-um-framework-unificado---parte-i%3A-simulacao-em-baixa-frequen).
In: L Simpósio Brasileiro de Pesquisa Operacional, Rio de Janeiro, Brazil.

## Features

Current features:
* Square-root Kalman filter and smoother
* Maximum likelihood estimation
* Forecasting
* Monte Carlo simulation
* Multivariate modeling
* User-defined models (input any `Z`, `T`, and `R`)
* Several pre-defined models, including:
  1. Basic structural model (trend, slope, seasonal)
  2. Structural model with exogenous variables
  3. Linear trend model
  4. Local level model
* Completion of missing values

Future features (work in progress):
* Exact initialization of Kalman filter

[build-img]: https://travis-ci.org/LAMPSPUC/StateSpaceModels.jl.svg?branch=master
[build-url]: https://travis-ci.org/LAMPSPUC/StateSpaceModels.jl

[codecov-img]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl?branch=master
