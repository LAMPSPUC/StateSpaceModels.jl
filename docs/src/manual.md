# Manual

## Introduction

In this package we consider the following state space model

```math
\begin{gather*}
    \begin{aligned}
        y_t &= Z_t \alpha_t  + \varepsilon_t, \quad \quad \quad t = 1 \dots n, \\
        \alpha_{t+1} &= T_t \alpha_t + R_t \eta_t,
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
        H_t & 0 & 0\\
        0 & Q_t & 0\\
        0 & 0 & P_1\\
    \end{bmatrix}
\end{pmatrix}
```

## Data Structures

```@docs
StateSpaceDimensions
StateSpaceModel
StateSpaceCovariance
SmoothedState
FilterOutput
StateSpace
```

## Default models
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
structuralmodel
```

## Estimation
The model estimation is made using the function `statespace(y, s; X, nseeds)`. It receives as argument the time series `y` and the desired seasonality `s`. The user can input exogenous variables using optional argument `X` and specify the desired number of random seeds `nseeds` to perform the estimation.

<!-- TODO -->

## Simulation

Simulation is made using the function `simulate`. It receives as argument a `StateSpace` object, the number of steps ahead `N` and the number of scenarios to simulate `S`.

```julia
simulation = simulate(ss, N, S)
```

## Filters

<!-- TODO sqrt kalman filter, put recursion and reference -->
<!-- TODO bigkappa kalman filter, put recursion and reference -->

## Optimization methods

<!-- LBFGS put reference and Optim manual -->