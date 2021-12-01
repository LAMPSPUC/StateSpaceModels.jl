@doc raw"""
    LocalLevelExplanatory(y::Vector{Fl}, X::Matrix{Fl}) where Fl

A local level model with explanatory variables is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + X_{1,t} \cdot \beta_{1,t} + \dots + X_{n,t} \cdot \beta_{n,t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \xi_{t} &\xi_{t} \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \beta_{1,t+1} &= \beta_{1,t} &\tau_{1, t} \sim \mathcal{N}(0, \sigma^2_{\tau_{1}})\\
        \dots &= \dots\\
        \beta_{n,t+1} &= \beta_{n,t} &\tau_{n, t} \sim \mathcal{N}(0, \sigma^2_{\tau_{n}})\\\\
    \end{aligned}
\end{gather*}
```

# Example
```jldoctest
julia> model = LocalLevelExplanatory(rand(100), rand(100, 1))
LocalLevelExplanatory
```
"""
mutable struct LocalLevelExplanatory <: StateSpaceModel
    hyperparameters::HyperParameters
    system::LinearUnivariateTimeVariant
    results::Results
    exogenous::Matrix

    function LocalLevelExplanatory(y::Vector{Fl}, X::Matrix{Fl}) where Fl

        @assert length(y) == size(X, 1)

        num_observations = size(X, 1)
        num_exogenous = size(X, 2)
        m = num_exogenous + 1

        # Define system matrices
        Z = [vcat(ones(Fl, 1), X[t, :]) for t in 1:num_observations]
        T = [Matrix{Fl}(I, m, m) for _ in 1:num_observations]
        R = [vcat(one(Fl), zeros(Fl, m-1, 1)) for _ in 1:num_observations]
        d = [zero(Fl) for _ in 1:num_observations]
        c = [zeros(m) for _ in 1:num_observations]
        H = [one(Fl) for _ in 1:num_observations]
        Q = [ones(Fl, 1, 1) for _ in 1:num_observations]

        system = LinearUnivariateTimeVariant{Fl}(y, Z, T, R, d, c, H, Q)

        # Define hyperparameters names
        names = [["sigma2_ε", "sigma2_η"];["β_$i" for i in 1:num_exogenous]]
        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, system, Results{Fl}(), X)
    end
end

# Obligatory methods
function default_filter(model::LocalLevelExplanatory)
    Fl = typeof_model_elements(model)
    a1 = zeros(Fl, num_states(model)) 
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    steadystate_tol = Fl(1e-5)
    return UnivariateKalmanFilter(a1, P1, num_states(model), steadystate_tol)
end

function get_beta_name(model::LocalLevelExplanatory, i::Int)
    return model.hyperparameters.names[i + 2]
end

function initial_hyperparameters!(model::LocalLevelExplanatory)
    Fl = typeof_model_elements(model)
    observed_variance = variance_of_valid_observations(model.system.y)
    initial_hyperparameters = Dict{String,Fl}(
        "sigma2_ε" => observed_variance, "sigma2_η" => observed_variance
    )
    # This is an heuristic for a good approximation
    initial_exogenous = model.exogenous \ model.system.y
    for i in axes(model.exogenous, 2)
        initial_hyperparameters[get_beta_name(model, i)] = initial_exogenous[i]
    end
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::LocalLevelExplanatory)
    for i in axes(model.exogenous, 2)
        constrain_identity!(model, get_beta_name(model, i))
    end
    constrain_variance!(model, "sigma2_ε")
    constrain_variance!(model, "sigma2_η")
    return model
end

function unconstrain_hyperparameters!(model::LocalLevelExplanatory)
    for i in axes(model.exogenous, 2)
        unconstrain_identity!(model, get_beta_name(model, i))
    end
    unconstrain_variance!(model, "sigma2_ε")
    unconstrain_variance!(model, "sigma2_η")
    return model
end

function fill_model_system!(model::LocalLevelExplanatory)
    H = get_constrained_value(model, "sigma2_ε")
    fill_H_in_time(model, H)
    for t in 1:length(model.system.Q)
        model.system.Q[t][1] = get_constrained_value(model, "sigma2_η")
    end
    return model
end

function fill_H_in_time(model::LocalLevelExplanatory, H::Fl) where Fl
    return fill_system_matrice_with_value_in_time(model.system.H, H)
end

function reinstantiate(::LocalLevelExplanatory, y::Vector{Fl}, X::Matrix{Fl}) where Fl
    return LocalLevelExplanatory(y, X)
end

has_exogenous(::LocalLevelExplanatory) = true 
