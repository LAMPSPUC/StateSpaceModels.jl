# Manual

## Estimation
The model estimation is made using the function `statespace(y, s; X, nseeds)`. It receives as argument the timeseries and the desired seasonality `s`.

The user can input explanatory variables in an ```Array{Float64, 2}``` variable `X` and specify the desired number of seeds to perform the estimation `nseeds`.

```julia
ss = statespace(y, s; X = X, nseeds = nseeds)
```

## Simulation

Simulation is made using the function `simulate`. It receives as argument a `StateSpace` object, the number of steps ahead `N` and the number of scenarios to simulate `S`.

```julia
simulation = simulate(ss, N, S)
```

## Example