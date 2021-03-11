using CSV, DataFrames, Plots, Dates, Test, LinearAlgebra, Optim
include("/Users/guilhermebodin/Documents/StateSpaceModels.jl/src/StateSpaceModels.jl")

nile = CSV.File(StateSpaceModels.NILE) |> DataFrame
model = StateSpaceModels.UnobservedComponents(nile.flow)
@show "fit começa $(now())"
StateSpaceModels.fit!(model)
@show "fit termina $(now())"
@test StateSpaceModels.loglike(model) ≈ -632.5376 atol = 1e-5 rtol = 1e-5
kf = StateSpaceModels.kalman_filter(model)
att = StateSpaceModels.get_filtered_state(kf)
plot(nile.flow)
plot!(att)


## Outlier
include("/Users/guilhermebodin/Documents/StateSpaceModels.jl/src/StateSpaceModels.jl")
nile = CSV.File(StateSpaceModels.NILE) |> DataFrame
nile.flow[50] += 2000
model = StateSpaceModels.UnobservedComponents(nile.flow)
Fl = Float64
steadystate_tol = Fl(1e-5)
n_states = StateSpaceModels.num_states(model)
a1 = zeros(Fl, n_states)
P1 = Fl(1e6) .* Matrix{Fl}(I, n_states, n_states)
lambda_o = 1.0
lambda_m = zero(Fl)
filt =  StateSpaceModels.RobustKalmanFilter(a1, P1, n_states, steadystate_tol, lambda_o, lambda_m)
StateSpaceModels.fit!(model; filter = filt, optimizer = StateSpaceModels.Optimizer(Optim.LBFGS(), Optim.Options(;show_trace = true)))
StateSpaceModels.loglike(model; filter = filt)
kf = StateSpaceModels.kalman_filter(model; filter = filt)
att = StateSpaceModels.get_filtered_state(kf)
plot(nile.flow)
plot!(att)
