# StateSpaceModels.jl

| **DOI** | **Build Status** | **Coverage** | **Documentation** |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1240453.svg)](https://doi.org/10.5281/zenodo.1240453) | [![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url] |[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://lampspuc.github.io/StateSpaceModels.jl/latest/)

## Installation

This package is registered in METADATA so you will can do `Pkg.add` it as follows:
```julia
pkg> add StateSpaceModels
```

## Notes

StateSpaceModels.jl is a package for modeling, forecasting and simulating time series in a state space framework based on structural models of Harvey (1989), in which a time series is decomposed in trend, slope and seasonals. Implementations were made based on the book "Time Series Analysis by State Space Methods" (2012) by J. Durbin and S. J. Koopman.

Work using this package:

[Simulating Low and High-Frequency Energy
Demand Scenarios in a Unified Framework – Part
I: Low-Frequency Simulation](https://proceedings.science/sbpo/papers/simulando-cenarios-de-demanda-em-baixa-e-alta-frequencia-em-um-framework-unificado---parte-i%3A-simulacao-em-baixa-frequen).
In: L Simpósio Brasileiro de Pesquisa Operacional, Rio de Janeiro, Brazil.

## Features

Current features:
* Basic structural model (trend, slope, seasonal)
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

[build-img]: https://travis-ci.org/LAMPSPUC/StateSpaceModels.jl.svg?branch=master
[build-url]: https://travis-ci.org/LAMPSPUC/StateSpaceModels.jl

[codecov-img]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl?branch=master
