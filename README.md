# StateSpaceModels.jl

StateSpaceModels.jl is a package for modeling, forecasting and simulating time series in a state space framework.

Estimation is done through function `statespace` and will automatically be run in parallel with all the currently active threads.

Simulation of future scenarios is done through function `simulate_statespace`.


Current features:
* Basic structural model (level, slope, seasonal)
* Square-root Kalman Filter and smoother
* Big Kappa initialization
* Monte Carlo simulation
* Parallel MLE estimation

Future features (work in progress):
* User-defined model
* Forecasting and confidence intervals
* Missing values completion
* Structural break
* Exact initialization
