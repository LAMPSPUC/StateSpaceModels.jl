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

### Kalman Filter

The implementation of the Kalman Filter follows the recursion

```math
\begin{gather*}
    \begin{aligned}
        v_t &= y_t - Z_t a_t,  &F_t &= Z_t P_t Z^{\top}_t + H\\
        a_{t|t} &= a_t - P_t Z^{\top}_t F_t^{-1} v_t, \quad \quad \quad &P_{t|t} &= P_t -  P_t Z^{\top}_t F_t^{-1}Z_t P_t\\
        a_{t+1} &= T a_t - K_t v_t, \quad \quad \quad &P_{t+1} &= T P_t(T - K_t Z_t)^{\top} + R Q R\\
    \end{aligned}
\end{gather*}
```
where ``K_t = T P_t Z^{\top}_t F_t^{-1}``. The terms ``a_{t+1}`` and ``P_{t+1}`` can be simplifed to 

```math
\begin{gather*}
    \begin{aligned}
        a_{t+1} &= T a_{t|t},\quad \quad \quad &P_{t+1} &= T P_{t|t} T^{\top} + R Q R\\
    \end{aligned}
\end{gather*}
```

In case of missing observation the mean of inovations ``v_t`` and variance of inovations ``F_t`` become `NaN` and the recursion becomes 

```math
\begin{gather*}
    \begin{aligned}
        a_{t|t} &= a_t, \quad \quad \quad &P_{t|t} &= P_t\\
        a_{t+1} &= T a_t, \quad \quad \quad &P_{t+1} &= T P_t T^{\top} + R Q R\\
    \end{aligned}
\end{gather*}
```

```@docs
StateSpaceModels.kalman_filter
```

The implementation of the its smoother follows the recursion


```math
\begin{gather*}
    \begin{aligned}
        r_{t-1} &= Z^{\top}_t F_t^{-1} v_t + L^{\top}_t r_t, \quad \quad  &N_{t-1} &= Z^{\top}_t F_t^{-1} Z_t + L^{\top}_t N_t L_t\\
        \alpha_t &= a_t + P_t r_t,  &V_t &= P_t - P_t N_{t-1} P_t  \\
    \end{aligned}
\end{gather*}
```
where ``L_t = T - K_t Z_t``.

In case of missing observation then ``r_{t-1}`` and ``N_{t-1}`` become 

```math
\begin{gather*}
    \begin{aligned}
        r_{t-1} &= T^{\top} r_t, \quad \quad  &N_{t-1} &= T^{\top}_t N_t T_t\\
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