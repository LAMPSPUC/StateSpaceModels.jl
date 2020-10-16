# StateSpaceModels.jl Documentation

StateSpaceModels.jl is a package for modeling, forecasting, and simulating time series in a state-space framework. Implementations were made based on the book "Time Series Analysis by State Space Methods" (2012) by James Durbin and Siem Jan Koopman. The notation of the variables in the code also aims to follow the book.

## Installation

This package is registered so you can simply `add` it using Julia's `Pkg` manager:
```julia
pkg> add StateSpaceModels
```

## Citing StateSpaceModels.jl

If you use StateSpaceModels.jl in your work, we kindly ask you to cite the following paper ([pdf](https://arxiv.org/pdf/1908.01757.pdf)):

    @article{SaavedraBodinSouto2019,
    title={StateSpaceModels.jl: a Julia Package for Time-Series Analysis in a State-Space Framework},
    author={Raphael Saavedra and Guilherme Bodin and Mario Souto},
    journal={arXiv preprint arXiv:1908.01757},
    year={2019}
    }
