# Examples

In this section we show examples of applications and different use cases of the package.

## Nile river annual flow

In this example we will follow some examples illustrated on Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. The data set consists of a series of readings of the annual flow volume at Aswan from 1871 to 1970.

```@setup nile
using StateSpaceModels
using Plots
using Dates
```

```@example nile
using CSV, DataFrames
nile = DataFrame!(CSV.File(StateSpaceModels.NILE))
plt = plot(nile.year, nile.flow, label = "Nile river annual flow")
```

We can fit the model

```@example nile
model = LocalLevel(nile.flow)
fit(model)
```

Analyse the filtered estimates for the level of the annual flow volume from the Kalman filter algorithm.

```@example nile
filter_output = kalman_filter(model)
plot!(plt, nile.year, filtered_estimates(filter_output), label = "Filtered level")
```

And analyse the smoothed estimates for the level of the annual flow volume from the Kalman smoother algorithm.

```@example nile
smoother_output = kalman_smoother(model)
plot!(plt, nile.year, smoothed_estimates(smoother_output), label = "Smoothed level")
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

StateSpaceModels.jl also enables generating scenarios for the forecasting horizon

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
fit(model)
```

And you have the exact same code for filtering and smoothing 

```@example nile
filter_output = kalman_filter(model)
smoother_output = kalman_smoother(model)
plot!(plt, nile.year, filtered_estimates(filter_output), label = "Filtered level")
plot!(plt, nile.year, smoothed_estimates(smoother_output), label = "Smoothed level")
```
