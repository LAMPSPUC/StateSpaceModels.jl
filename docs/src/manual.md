# Manual

## Introduction

In this package we consider the following state-space model

```math
\begin{gather*}
    \begin{aligned}
        y_t &= Z_t \alpha_t  + \varepsilon_t, \quad \quad \quad t = 1 \dots n, \\
        \alpha_{t+1} &= T \alpha_t + R \eta_t,
    \end{aligned}
\end{gather*}
```
where
```math
\begin{bmatrix}
    \varepsilon_t \\
    \eta_t \\
    \alpha_1
\end{bmatrix}
\sim
NID
\begin{pmatrix}
    \begin{bmatrix}
        0 \\
        0 \\
        a_1
    \end{bmatrix}
    ,
    \begin{bmatrix}
        H & 0 & 0\\
        0 & Q & 0\\
        0 & 0 & P_1\\
    \end{bmatrix}
\end{pmatrix}
```

## Data structures

```@docs
StateSpaceDimensions
StateSpaceModel
StateSpaceCovariance
SmoothedState
FilterOutput
StateSpace
```

## Predefined models
The local level model is defined by

```math
\begin{gather*}
    \begin{aligned}
        y_t &=  \mu_t  + \varepsilon_t \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_t + \eta_t \quad \eta_t \sim \mathcal{N}(0, \sigma^2_{\eta})\\
    \end{aligned}
\end{gather*}
```

```@docs
local_level
```

The linear trend model is defined by

```math
\begin{gather*}
    \begin{aligned}
        y_t &=  \mu_t  + \varepsilon_t \quad &\varepsilon_t \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_t + \nu_t + \xi_t \quad &\xi_t \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \nu_{t+1} &= \nu_t + \zeta_t \quad &\zeta_t \sim \mathcal{N}(0, \sigma^2_{\zeta})\\
    \end{aligned}
\end{gather*}
```

```@docs
linear_trend
```

The structural model is defined by

<!-- TODO mathematical model -->
```@docs
structural
```

## Estimation
The model estimation is made using the function `statespace(model; filter_type = KalmanFilter, optimization_method = RandomSeedsLBFGS(), verbose = 1)`. It receives as argument the pre-specified `StateSpaceModel` object `model`. Optionally, the user can define the Kalman filter variant to be used, the optimization method and the verbosity level.

```@docs
statespace
```

## Forecasting

Forecasting is conducted with the function `forecast`. It receives as argument a `StateSpace` object and the number of steps ahead `N`.

```@docs
forecast
```

## Simulation

Simulation is made using the function `simulate`. It receives as argument a `StateSpace` object, the number of steps ahead `N` and the number of scenarios to simulate `S`.

```@docs
simulate
```

## Filters

### Interface

StateSpaceModels has an interface that allows users to define their own versions of the kalman filter.

```@docs
StateSpaceModels.AbstractFilter
StateSpaceModels.AbstractSmoother
kalman_filter_and_smoother
```

every filter must provide its own version of the `statespace_likelihood` function
```@docs
StateSpaceModels.statespace_likelihood
```


### Kalman Filter

```@docs
StateSpaceModels.kalman_filter
StateSpaceModels.smoother
```

### Square Root Kalman Filter

```@docs
StateSpaceModels.sqrt_kalman_filter
StateSpaceModels.sqrt_smoother
StateSpaceModels.filtered_state
```

## Optimization methods

```@docs
StateSpaceModels.AbstractOptimizationMethod
StateSpaceModels.estimate_statespace
```

### RandomSeedsLBFGS

```@docs
StateSpaceModels.RandomSeedsLBFGS
```