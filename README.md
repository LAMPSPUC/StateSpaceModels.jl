[build-img]: https://github.com/LAMPSPUC/StateSpaceModels.jl/workflows/CI/badge.svg?branch=master
[build-url]: https://github.com/LAMPSPUC/StateSpaceModels.jl/actions?query=workflow%3ACI

[codecov-img]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/gh/LAMPSPUC/StateSpaceModels.jl?branch=master

# StateSpaceModels.jl

| **Build Status** | **Coverage** | **Documentation** |
|:-----------------:|:-----------------:|:-----------------:|
| [![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url] |[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://lampspuc.github.io/StateSpaceModels.jl/latest/)

StateSpaceModels.jl is a package for modeling, forecasting, and simulating time series in a state-space framework. Implementations were made based on the book "Time Series Analysis by State Space Methods" (2012) by James Durbin and Siem Jan Koopman. The notation of the variables in the code also follows the book.

## Quickstart
```julia
import Pkg

Pkg.add("StateSpaceModels")

using StateSpaceModels

y = randn(100)

model = LocalLevel(y)

fit!(model)

results(model)

forecast(model, 10)

kf = kalman_filter(model)

v = get_innovations(kf)

ks = kalman_smoother(model)

alpha = get_smoothed_state(ks)
```

## Features

Current features include:
* Kalman filter and smoother
* Maximum likelihood estimation
* Forecasting and Monte Carlo simulation
* User-defined models (user specifies the state-space system)
* Several predefined models, including:
  * Exponential Smoothing (ETS, all the linear ones)
  * Unobserved components (local level, basic structural, ...)
  * SARIMA
  * Linear regression
  * Naive models
* Completion of missing values
* Diagnostics for the residuals of fitted models
* Visualization recipes

## Contributing

* PRs such as adding new models and fixing bugs are very welcome!
* For nontrivial changes, you'll probably want to first discuss the changes via issue.

## Citing StateSpaceModels.jl

If you use StateSpaceModels.jl in your work, we kindly ask you to cite the [following paper](https://arxiv.org/abs/1908.01757):

    @article{SaavedraBodinSouto2019,
      title={StateSpaceModels.jl: a Julia Package for Time-Series Analysis in a State-Space Framework},
      author={Raphael Saavedra and Guilherme Bodin and Mario Souto},
      journal={arXiv preprint arXiv:1908.01757},
      year={2019}
    }
