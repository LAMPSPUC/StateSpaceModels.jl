@doc raw"""
    LocalLevelExplanatoryTimeVarying(y::Vector{Fl}, X::Matrix{Fl}) where Fl

A local level model allowing time-varying regressors is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + X_{1,t} \cdot \beta_{1,t} + \dots + X_{n,t} \cdot \beta_{n,t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \xi_{t} &\xi_{t} \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \beta_{1,t+1} &= \beta_{1,t} + \tau_{1, t} &\tau_{1, t} \sim \mathcal{N}(0, \sigma^2_{\tau_{1}})\\
        \dots &= \dots\\
        \beta_{n,t+1} &= \beta_{n,t} + \tau_{n, t} &\tau_{n, t} \sim \mathcal{N}(0, \sigma^2_{\tau_{n}})\\\\
    \end{aligned}
\end{gather*}
```

This model can quickly overfit, so it is useful to fix the variance of one or more of the tau hyperparamters at zero - with all fixed to zero, this model becomes the LocalLevelExplanatory
# Example
```jldoctest
julia> model = LocalLevelExplanatoryTimeVarying(rand(100), rand(100, 1))
LocalLevelExplanatoryTimeVarying
```
"""
mutable struct LocalLevelExplanatoryTimeVarying <: StateSpaceModel
    hyperparameters::HyperParameters
    system::LinearUnivariateTimeVariant
    results::Results
    exogenous::Matrix

    function LocalLevelExplanatoryTimeVarying(y::Vector{Fl}, X::Matrix{Fl}) where Fl

        @assert length(y) == size(X, 1)

        num_observations = size(X, 1)
        num_exogenous = size(X, 2)
        m = num_exogenous + 1

        # Define system matrices
        Z = [vcat(ones(Fl, 1), X[t, :]) for t in 1:num_observations]
        T = [Matrix{Fl}(I, m, m) for _ in 1:num_observations]
        R = [Matrix{Fl}(I, m, m) for _ in 1:num_observations]
        d = [zero(Fl) for _ in 1:num_observations]
        c = [zeros(m) for _ in 1:num_observations]
        H = [one(Fl) for _ in 1:num_observations]
        Q = [Matrix{Fl}(I, m, m) for _ in 1:num_observations]

        system = LinearUnivariateTimeVariant{Fl}(y, Z, T, R, d, c, H, Q)

        # Define hyperparameters names
        names = [["sigma2_ε", "sigma2_η"];["β_$i" for i in 1:num_exogenous]; ["tau2_$i" for i in 1:num_exogenous]]
        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, system, Results{Fl}(), X)
    end
end

# Obligatory methods
function default_filter(model::LocalLevelExplanatoryTimeVarying)
    Fl = typeof_model_elements(model)
    a1 = zeros(Fl, num_states(model)) 
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    steadystate_tol = Fl(1e-5)
    return UnivariateKalmanFilter(a1, P1, num_states(model), steadystate_tol)
end

function get_beta_name(model::LocalLevelExplanatoryTimeVarying, i::Int)
    return model.hyperparameters.names[i + 2]
end

function initial_hyperparameters!(model::LocalLevelExplanatoryTimeVarying)
    #Fill all with observed variance - betas redone later, taus are pretty bad
    Fl = typeof_model_elements(model)
    observed_variance = var(model.system.y[findall(!isnan, model.system.y)])
    initial_hyperparameters = Dict{String,Fl}(
        model.hyperparameters.names .=> observed_variance
    )
    # This is an heuristic for a good approximation
    initial_exogenous = model.exogenous \ model.system.y
    for i in axes(model.exogenous, 2)
        initial_hyperparameters[get_beta_name(model, i)] = initial_exogenous[i]
    end
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::LocalLevelExplanatoryTimeVarying)
    betas = [get_beta_name(model, i) for i in axes(model.exogenous, 2)]
    constrain_identity!.(Ref(model), betas)

    non_betas = setdiff(model.hyperparameters.names, betas)
    constrain_variance!.(Ref(model), non_betas)
    return model
end

function unconstrain_hyperparameters!(model::LocalLevelExplanatoryTimeVarying)
    betas = [get_beta_name(model, i) for i in axes(model.exogenous, 2)]
    unconstrain_identity!.(Ref(model), betas)

    non_betas = setdiff(model.hyperparameters.names, betas)
    unconstrain_variance!.(Ref(model), non_betas)
    return model
end

function fill_model_system!(model::LocalLevelExplanatoryTimeVarying)
    H = get_constrained_value(model, "sigma2_ε")
    fill_H_in_time(model, H)

    numtaus = Int((length(model.hyperparameters.names) - 2) / 2)
    taus = model.hyperparameters.names[Int(end - numtaus + 1): end]

    constrained_values_taus = get_constrained_value.(Ref(model), taus)
    
    for t in 1:length(model.system.Q)
        model.system.Q[t] = diagm([get_constrained_value(model, "sigma2_η"); constrained_values_taus])
    end
    return model
end

function fill_H_in_time(model::LocalLevelExplanatoryTimeVarying, H::Fl) where Fl
    return fill_system_matrice_with_value_in_time(model.system.H, H)
end

function reinstantiate(::LocalLevelExplanatoryTimeVarying, y::Vector{Fl}, X::Matrix{Fl}) where Fl
    return LocalLevelExplanatoryTimeVarying(y, X)
end

has_exogenous(::LocalLevelExplanatoryTimeVarying) = true 
