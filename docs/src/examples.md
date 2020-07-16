# Examples

In this section we show examples of applications and different use cases of the package.

## Nile river annual flow

In this example we will follow some examples illustrated on Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. The data set consists of a series of readings of the annual flow volume at Aswan from 1871 to 1970.

```@setup nile
using StateSpaceModels
using Plots
```

```@example nile
plot(StateSpaceModels.NILE, label = "Annual nile river flow")
```