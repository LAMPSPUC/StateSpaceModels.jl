# Examples

In this section we show examples of applications and different use cases of the package.

## Nile river annual flow

In this example we will follow some examples illustrated on Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. The data set consists of a series of readings of the annual flow volume at Aswan from 1871 to 1970.

```@setup nile
using StateSpaceModels
using Plots
```

```@example nile
using CSV, DataFrames
nile = DataFrame!(CSV.File(StateSpaceModels.NILE))
plt = plot(nile.year, nile.flow, label = "Annual nile river flow")
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