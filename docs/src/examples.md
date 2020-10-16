# Examples

In this section we show examples of applications and use cases of the package.

## Nile river annual flow

Here we will follow an example from Durbin & Koopman's book.
We will use the [`LocalLevel`](@ref) model applied to the annual flow of the Nile river at the city of Aswan between 1871 and 1970.

```@setup nile
using StateSpaceModels
using Plots
using Dates
```

First, we load the data:

```@example nile
using CSV, DataFrames
nile = CSV.read(StateSpaceModels.NILE, DataFrame)
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

StateSpaceModels.jl can also be used to obtain forecasts. 
Here we forecast 10 steps ahead:

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

The package also handles missing values automatically. 
To that end, the package considers that any `NaN` entries in the observations are missing values.

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

TODO

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
using CSV, DataFrames
df = CSV.read(StateSpaceModels.VEHICLE_FATALITIES, DataFrame)
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




