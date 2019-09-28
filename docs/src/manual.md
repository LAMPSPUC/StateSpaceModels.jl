# Manual

## Introduction

In this package we consider the following state-space model

```math
\begin{gather*}
    \begin{aligned}
        y_{t} &= Z_{t} \alpha_{t}  + \varepsilon_{t}, \quad \quad \quad t = 1 \dots n, \\
        \alpha_{t+1} &= T \alpha_{t} + R \eta_{t},
    \end{aligned}
\end{gather*}
```
where
```math
\begin{bmatrix}
    \varepsilon_{t} \\
    \eta_{t} \\
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
        y_{t} &=  \mu_{t}  + \varepsilon_{t} \quad \varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \eta_{t} \quad \eta_{t} \sim \mathcal{N}(0, \sigma^2_{\eta})\\
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
        y_{t} &=  \mu_{t}  + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \nu_{t} + \xi_{t} \quad &\xi_{t} \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \nu_{t+1} &= \nu_{t} + \zeta_{t} \quad &\zeta_{t} \sim \mathcal{N}(0, \sigma^2_{\zeta})\\
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

### Kalman Filter

The implementation of the Kalman Filter follows the recursion

```math
\begin{gather*}
    \begin{aligned}
        v_{t} &= y_{t} - Z_{t} a_{t},  &F_{t} &= Z_{t} P_{t} Z^{\top}_{t} + H\\
        a_{t|t} &= a_{t} - P_{t} Z^{\top}_{t} F_{t}^{-1} v_{t}, \quad \quad \quad &P_{t|t} &= P_{t} -  P_{t} Z^{\top}_{t} F_{t}^{-1}Z_{t} P_{t}\\
        a_{t+1} &= T a_{t} - K_{t} v_{t}, \quad \quad \quad &P_{t+1} &= T P_{t}(T - K_{t} Z_{t})^{\top} + R Q R\\
    \end{aligned}
\end{gather*}
```
where ``K_{t} = T P_{t} Z^{\top}_{t} F_{t}^{-1}``. The terms ``a_{t+1}`` and ``P_{t+1}`` can be simplifed to 

```math
\begin{gather*}
    \begin{aligned}
        a_{t+1} &= T a_{t|t},\quad \quad \quad &P_{t+1} &= T P_{t|t} T^{\top} + R Q R\\
    \end{aligned}
\end{gather*}
```

In case of missing observation the mean of inovations ``v_{t}`` and variance of inovations ``F_{t}`` become `NaN` and the recursion becomes 

```math
\begin{gather*}
    \begin{aligned}
        a_{t|t} &= a_{t}, \quad \quad \quad &P_{t|t} &= P_{t}\\
        a_{t+1} &= T a_{t}, \quad \quad \quad &P_{t+1} &= T P_{t} T^{\top} + R Q R\\
    \end{aligned}
\end{gather*}
```

```@docs
StateSpaceModels.kalman_filter
```

The implementation of the smoother follows the recursion


```math
\begin{gather*}
    \begin{aligned}
        r_{t-1} &= Z^{\top}_{t} F_{t}^{-1} v_{t} + L^{\top}_{t} r_{t}, \quad \quad  &N_{t-1} &= Z^{\top}_{t} F_{t}^{-1} Z_{t} + L^{\top}_{t} N_{t} L_{t}\\
        \alpha_{t} &= a_{t} + P_{t} r_{t},  &V_{t} &= P_{t} - P_{t} N_{t-1} P_{t}  \\
    \end{aligned}
\end{gather*}
```
where ``L_{t} = T - K_{t} Z_{t}``.

In case of missing observation then ``r_{t-1}`` and ``N_{t-1}`` become 

```math
\begin{gather*}
    \begin{aligned}
        r_{t-1} &= T^{\top} r_{t}, \quad \quad  &N_{t-1} &= T^{\top}_{t} N_{t} T_{t}\\
    \end{aligned}
\end{gather*}
```

```@docs
StateSpaceModels.smoother
```

### Square Root Kalman Filter

```@docs
StateSpaceModels.sqrt_kalman_filter
StateSpaceModels.sqrt_smoother
StateSpaceModels.filtered_state
```

## Filter interface

StateSpaceModels has an interface that allows users to define their own versions of the Kalman filter.

```@docs
StateSpaceModels.AbstractFilter
StateSpaceModels.AbstractSmoother
kfas
```

Every filter must provide its own version of the `statespace_likelihood` function
```@docs
StateSpaceModels.statespace_likelihood
```

## Optimization methods

### RandomSeedsLBFGS

```@docs
StateSpaceModels.RandomSeedsLBFGS
```

## Optimization methods interface

```@docs
StateSpaceModels.AbstractOptimizationMethod
StateSpaceModels.estimate_statespace
```

## Diagnostics
After the estimation is completed, diagnostics can be run over the residuals. The implemented diagnostics include the Jarque-Bera normality test, the Ljung-Box independence test, and a homoscedasticity test.

```@docs
diagnostics
StateSpaceModels.residuals
StateSpaceModels.jarquebera
StateSpaceModels.ljungbox
StateSpaceModels.homoscedast
```