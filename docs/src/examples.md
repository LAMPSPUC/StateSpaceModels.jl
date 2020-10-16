# Examples

In this section we show examples of applications and different use cases of the package.

## Nile river annual flow

In this example we will follow what is illustrated on Durbin, James, & Siem Jan Koopman. (2012). 
"Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. We will 
study the [`LocalLevel`](@ref) model with a series of readings of the annual flow volume at 
Aswan from 1871 to 1970.

```@setup nile
using StateSpaceModels
using Plots
using Dates
```

```@example nile
using CSV, DataFrames
nile = CSV.read(StateSpaceModels.NILE, DataFrame)
plt = plot(nile.year, nile.flow, label = "Nile river annual flow")
```

We can fit a [`LocalLevel`](@ref) model

```@example nile
model = LocalLevel(nile.flow)
fit!(model)
```

Analyse the filtered estimates for the level of the annual flow volume from the Kalman filter algorithm.

```@example nile
filter_output = kalman_filter(model)
plot!(plt, nile.year, get_filtered_state(filter_output), label = "Filtered level")
```

And analyse the smoothed estimates for the level of the annual flow volume from the Kalman smoother algorithm.

```@example nile
smoother_output = kalman_smoother(model)
plot!(plt, nile.year, get_smoothed_state(smoother_output), label = "Smoothed level")
```

StateSpaceModels.jl has a flexible forecasting schema that easily allows users to 
get forecasts and confidence intervals. Here we forecast 10 steps ahead.

```@example nile
steps_ahead = 10
dates = collect(nile.year[end] + Year(1):Year(1):nile.year[end] + Year(10))
forec = forecast(model, 10)
expected_value = forecast_expected_value(forec)
plot!(plt, dates, expected_value, label = "Forecast")
```

StateSpaceModels.jl also enables simulating multiple scenarios for the forecasting horizon based on the estimated distributions.

```@example nile
scenarios = simulate_scenarios(model, 10, 100)
plot!(plt, dates, scenarios[:, 1, :], label = "", color = "grey", width = 0.2)
```

StateSpaceModels.jl handles missing values automatically, the package consider that observations with `NaN` are missing values.

```@example nile
nile.flow[[collect(21:40); collect(61:80)]] .= NaN
plt = plot(nile.year, nile.flow, label = "Annual nile river flow")
```

We can proceed to the same analysis

```@example nile
model = LocalLevel(nile.flow)
fit!(model)
```

And you have the exact same code for filtering and smoothing 

```@example nile
filter_output = kalman_filter(model)
smoother_output = kalman_smoother(model)
plot!(plt, nile.year, get_filtered_state(filter_output), label = "Filtered level")
plot!(plt, nile.year, get_smoothed_state(smoother_output), label = "Smoothed level")
```

## Log of airline passengers

## Finland road traffic fatalities

In this example we will follow what is illustrated on Commandeur, Jacques J.F. & Koopman, 
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




