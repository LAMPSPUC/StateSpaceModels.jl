# StateSpaceModels.jl Documentation

StateSpaceModels.jl is a package for modeling, forecasting, and simulating time series in a state-space framework. Implementations were made based on the book "Time Series Analysis by State Space Methods" (2012) by James Durbin and Siem Jan Koopman. The notation of the variables in the code also follows the book.

## Installation

This package is registered in METADATA so you can `Pkg.add` it as follows:
```julia
pkg> add StateSpaceModels
```

## Features

Current features:
* Kalman filter and smoother
* Square-root filter and smoother
* Maximum likelihood estimation
* Forecasting
* Monte Carlo simulation
* Multivariate modeling
* User-defined models (input any `Z`, `T`, and `R`)
* Several predefined models, including:
  1. Basic structural model (trend, slope, seasonal)
  2. Structural model with exogenous variables
  3. Linear trend model
  4. Local level model
* Completion of missing values

Planned features:
* Exact initialization of the Kalman filter
* EM algorithm for maximum likelihood estimation
* Univariate treatment of multivariate models

## Works

Works using this package:

[Simulating Low and High-Frequency Energy
Demand Scenarios in a Unified Framework – Part
I: Low-Frequency Simulation](https://proceedings.science/sbpo/papers/simulando-cenarios-de-demanda-em-baixa-e-alta-frequencia-em-um-framework-unificado---parte-i%3A-simulacao-em-baixa-frequen).
In: L Simpósio Brasileiro de Pesquisa Operacional, Rio de Janeiro, Brazil.
