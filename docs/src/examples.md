# Examples

In this section we show examples of applications and use cases of the package.

## Nile river annual flow

Here we will follow an example from Durbin & Koopman's book. We will use the [`LocalLevel`](@ref) model applied to the annual flow of the Nile river at the city of Aswan between 1871 and 1970.

```@setup nile
using StateSpaceModels
using Plots
using Dates
```

First, we load the data:

```@example nile
using CSV, DataFrames
nile = CSV.File(StateSpaceModels.NILE) |> DataFrame
plt = plot(nile.year, nile.flow, label = "Nile river annual flow")
```

Next, we fit a [`LocalLevel`](@ref) model:

```@example nile
model = LocalLevel(nile.flow)
fit!(model)
```

We can analyze the filtered estimates for the level of the annual flow:

```@example nile
filter_output = kalman_filter(model)
plot!(plt, nile.year, get_filtered_state(filter_output), label = "Filtered level")
```

We can do the same for the smoothed estimates for the level of the annual flow:

```@example nile
smoother_output = kalman_smoother(model)
plot!(plt, nile.year, get_smoothed_state(smoother_output), label = "Smoothed level")
```

StateSpaceModels.jl can also be used to obtain forecasts. Here we forecast 10 steps ahead:

```@example nile
steps_ahead = 10
dates = collect(nile.year[end] + Year(1):Year(1):nile.year[end] + Year(10))
forec = forecast(model, 10)
expected_value = forecast_expected_value(forec)
plot!(plt, dates, expected_value, label = "Forecast")
```

We can also simulate multiple scenarios for the forecasting horizon based on the estimated predictive distributions.

```@example nile
scenarios = simulate_scenarios(model, 10, 100)
plot!(plt, dates, scenarios[:, 1, :], label = "", color = "grey", width = 0.2)
```

The package also handles missing values automatically. To that end, the package considers that any `NaN` entries in the observations are missing values.

```@example nile
nile.flow[[collect(21:40); collect(61:80)]] .= NaN
plt = plot(nile.year, nile.flow, label = "Annual nile river flow")
```

Even though the series has several missing values, the same analysis is possible:

```@example nile
model = LocalLevel(nile.flow)
fit!(model)
```

And the exact same code can be used for filtering and smoothing:

```@example nile
filter_output = kalman_filter(model)
smoother_output = kalman_smoother(model)
plot!(plt, nile.year, get_filtered_state(filter_output), label = "Filtered level")
plot!(plt, nile.year, get_smoothed_state(smoother_output), label = "Smoothed level")
```

## Airline passengers

We often write the model SARIMA model as an ARIMA ``(p,d,q) \times (P,D,Q,s)``, where the lowercase letters indicate the specification for the non-seasonal component, and the uppercase letters indicate the specification for the seasonal component; ``s`` is the periodicity of the seasons (e.g. it is often 4 for quarterly data or 12 for monthly data). The data process can be written generically as

```math
\begin{equation}
    \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D y_t = A(t) + \theta_q (L) \tilde \theta_Q (L^s) \epsilon_t
\end{equation}
```

where
 * ``\phi_p (L)`` is the non-seasonal autoregressive lag polynomial,
 * ``\tilde \phi_P (L^s)`` is the seasonal autoregressive lag polynomial,
 * ``\Delta^d \Delta_s^D y_t`` is the time series, differenced ``d`` times, and seasonally differenced ``D`` times.,
 * ``A(t)`` is the trend polynomial (including the intercept),
 * ``\theta_q (L)`` is the non-seasonal moving average lag polynomial,
 * ``\tilde \theta_Q (L^s)`` is the seasonal moving average lag polynomial

sometimes we rewrite this as:

```math
\begin{equation}
    \phi_p (L) \tilde \phi_P (L^s) y_t^* = A(t) + \theta_q (L) \tilde \theta_Q (L^s) \epsilon_t
\end{equation}
```

where ``y_t^* = \Delta^d \Delta_s^D y_t``. This emphasizes that just as in the simple case, after we take differences (here both non-seasonal and seasonal) to make the data stationary, the resulting model is just an ARMA model.

As an example, consider the airline model ARIMA ``(0,1,1) \times (0,1,1,12)``. The data process can be written in the form above as:,

```math
\begin{equation}
    \Delta \Delta_{12} y_t = (1 - \theta_1 L) (1 - \tilde \theta_1 L^{12}) \epsilon_t
\end{equation}
```

Here, we have:

 * ``\phi_p (L) = 1``, (i.e. there is no auto regressive effect)
 * ``\tilde \phi_P (L^s) = 1``, (i.e. there is no seasonal auto regressive effect)
 * ``d = 1, D = 1, s=12`` indicating that ``y_t^*`` is derived from ``y_t`` by taking first-differences and then taking 12-th differences.,
 * ``A(t) = 0`` no trend,
 * ``\theta_q (L) = (1 - \theta_1 L)``,
 * ``\tilde \theta_Q (L^s) = (1 - \tilde \theta_1 L^{12})``,

It may still be confusing to see the two lag polynomials in front of the error variable, but notice that we can multiply the lag polynomials together to get the following model:

```math
\begin{equation}
    y_t^* = (1 - \theta_1 L - \tilde \theta_1 L^{12} + \theta_1 \tilde \theta_1 L^{13}) \epsilon_t
\end{equation}
```

For the airline model ARIMA ``(0,1,1) \times (0,1,1,12)`` with an intercept, the command is:

```@setup airline
using StateSpaceModels, CSV, DataFrames
using Plots
```

```@example airline
using CSV, DataFrames
air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(air_passengers.passengers)
model = SARIMA(log_air_passengers; order = (0, 1, 1), seasonal_order = (0, 1, 1, 12))
fit!(model)
print_results(model)
```

To make a forecast of 24 steps ahead of the model the command is:
```@example airline
forec = forecast(model, 24)
plot(model, forec; legend = :topleft)
```
> The text from this example is based on Python`s [statsmodels](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html) library.
> The estimates for this example match up to the 3th decimal place the results of the paper [State Space Methods in Ox/SsfPack](https://www.jstatsoft.org/article/view/v041i03) from the journal of statistical software.

## Finland road traffic fatalities

In this example, we will follow what is illustrated on Commandeur, Jacques J.F. & Koopman, 
Siem Jan, 2007. "An Introduction to State Space Time Series Analysis," OUP Catalogue, 
Oxford University Press (Chapter 3). We will study the [`LocalLinearTrend`](@ref) model with 
a series of the log of road traffic fatalities in Finalnd and analyse its slope to tell if
the trend of fatalities was increasing or decrasing during different periods of time.

```@setup fatalities
using StateSpaceModels
using Plots
```

```@example fatalities
using StateSpaceModels, Plots, CSV, DataFrames
df = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
log_ff = log.(df.ff)
plt = plot(df.date, log_ff, label = "Log of Finland road traffic fatalities")
```

We fit a [`LocalLinearTrend`](@ref)

```@example fatalities
model = LocalLinearTrend(log_ff)
fit!(model)
```

By extracting the smoothed slope we conclude that according to our model the trend of fatalities 
in Finland was increasing in the years 1970, 1982, 1984 through to 1988, and in 1998

```@example fatalities
smoother_output = kalman_smoother(model)
plot(df.date, get_smoothed_state(smoother_output)[:, 2], label = "slope")
```

## Vehicle tracking

This example illustrates how to perform vehicle tracking from noisy data.

```@setup vehicle_tracking
using StateSpaceModels, Random
using Plots
```

```@example vehicle_tracking
using Random
Random.seed!(1)

# Define a random trajectory
n = 100
H = [1 0; 0 1.0]
Q = [1 0; 0 1.0]
rho = 0.1
model = VehicleTracking(rand(n, 2), rho, H, Q)
initial_state = [0.0, 0, 0, 0]
sim = StateSpaceModels.simulate(model.system, initial_state, n)

# Use a Kalman filter to get the predictive and filtered states
model = VehicleTracking(sim, 0.1, H, Q)
kalman_filter(model)
pos_pred = get_predictive_state(model)
pos_filtered = get_filtered_state(model)

# Plot a gif illustrating the result
using Plots
anim = @animate for i in 1:n
    plot(sim[1:i, 1], sim[1:i, 2], label="Measured position", line=:scatter, lw=2, markeralpha=0.2, color=:black, title="Vehicle tracking")
    plot!(pos_pred[1:i+1, 1], pos_pred[1:i+1, 3], label = "Predicted position", lw=2, color=:forestgreen)
    plot!(pos_filtered[1:i, 1], pos_filtered[1:i, 3], label = "Filtered position", lw=2, color=:indianred)
end
gif(anim, "anim_fps15.gif", fps = 15)
```

## Cross validation of the forecasts of a model

Often times users would like to compare the forecasting skill of different models. The function 
[`cross_validation`](@ref) makes it easy to make a rolling window scheme of estimations and forecasts 
that allow users to track each model forecasting skill per lead time. A simple plot recipe is
implemented to help users to interpret the results easily.

```@setup bt
using StateSpaceModels, CSV, DataFrames
using Plots
```

```@example bt
using CSV, DataFrames
air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(air_passengers.passengers)
model = BasicStructural(log_air_passengers, 12)
b = cross_validation(model, 24, 50)
plot(b, "Basic structural model")
```