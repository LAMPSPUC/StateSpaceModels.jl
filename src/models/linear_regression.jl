struct RegressionHyperParametersAuxiliary
    beta_names::Vector{String}
    function RegressionHyperParametersAuxiliary(num_states::Int)
        return new(beta_names(num_states))
    end
end

@doc raw"""
    LinearRegression(X::Matrix{Fl}, y::Vector{Fl}) where Fl

The linear regression state-space model is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  X_{1,t} \cdot \beta_{1,t} + \dots + X_{n,t} \cdot \beta_{n,t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \beta_{1,t+1} &= \beta_{1,t}\\
        \dots &= \dots\\
        \beta_{n,t+1} &= \beta_{n,t}\\
    \end{aligned}
\end{gather*}
```

# Example
```jldoctest
julia> model = LinearRegression(rand(100, 2), rand(100))
LinearRegression
```
"""
mutable struct LinearRegression <: StateSpaceModel
    hyperparameters::HyperParameters
    hyperparameters_auxiliary::RegressionHyperParametersAuxiliary
    system::LinearUnivariateTimeVariant
    results::Results
    exogenous::Matrix

    function LinearRegression(X::Matrix{Fl}, y::Vector{Fl}) where Fl

        # Treat possible input errors
        @assert length(y) == size(X, 1)

        num_observations = size(X, 1)
        num_exogenous = size(X, 2)

        Z = [X[t, :] for t in 1:num_observations]
        T = [Matrix{Fl}(I, num_exogenous, num_exogenous) for _ in 1:num_observations]
        R = [zeros(num_exogenous, 1) for _ in 1:num_observations]
        d = [zero(Fl) for _ in 1:num_observations]
        c = [zeros(num_exogenous) for _ in 1:num_observations]
        H = [one(Fl) for _ in 1:num_observations]
        Q = [zeros(Fl, 1, 1) for _ in 1:num_observations]

        system = LinearUnivariateTimeVariant{Fl}(y, Z, T, R, d, c, H, Q)

        names = [["β_$i" for i in 1:num_exogenous]; "sigma2_ε"]
        hyperparameters = HyperParameters{Fl}(names)

        hyperparameters_auxiliary = RegressionHyperParametersAuxiliary(num_states(system))

        return new(hyperparameters, hyperparameters_auxiliary, system, Results{Fl}(), X)
    end
end

function beta_names(num_states::Int)
    str = Vector{String}(undef, num_states)
    for i in eachindex(str)
        str[i] = "β_$i"
    end
    return str
end

function get_beta_name(model::LinearRegression, i::Int)
    return model.hyperparameters_auxiliary.beta_names[i]
end

function fill_H_in_time(model::LinearRegression, H::Fl) where Fl
    return fill_system_matrice_with_value_in_time(model.system.H, H)
end

# Obligatory functions
function default_filter(model::LinearRegression)
    Fl = typeof_model_elements(model)
    a1 = zeros(Fl, num_states(model))
    return RegressionKalmanFilter(a1)
end

function initial_hyperparameters!(model::LinearRegression)
    Fl = typeof_model_elements(model)
    initial_hyperparameters = Dict{String,Fl}("sigma2_ε" => var(model.system.y))
    # The optimal regressors are the result of X \ y
    betas = model.exogenous \ model.system.y
    for i in 1:num_states(model)
        initial_hyperparameters[get_beta_name(model, i)] = betas[i]
    end
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::LinearRegression)
    for i in 1:num_states(model)
        constrain_identity!(model, get_beta_name(model, i))
    end
    constrain_variance!(model, "sigma2_ε")
    return model
end

function unconstrain_hyperparameters!(model::LinearRegression)
    for i in 1:num_states(model)
        unconstrain_identity!(model, get_beta_name(model, i))
    end
    unconstrain_variance!(model, "sigma2_ε")
    return model
end

function fill_model_system!(model::LinearRegression)
    # Fill the same H for every timestamp
    H = get_constrained_value(model, "sigma2_ε")
    fill_H_in_time(model, H)
    return nothing
end

function fill_model_filter!(filter::KalmanFilter, model::LinearRegression)
    for i in axes(filter.kalman_state.a, 1)
        filter.kalman_state.a[i] = get_constrained_value(model, get_beta_name(model, i))
    end
    return filter
end

function reinstantiate(::LinearRegression, y::Vector{Fl}, X::Matrix{Fl}) where Fl
    return LinearRegression(X, y)
end

has_exogenous(::LinearRegression) = true
