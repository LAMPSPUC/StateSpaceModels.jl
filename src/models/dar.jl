@doc raw"""
    DAR(y::Vector{Fl}, lags::Int) where Fl

A Dynamic Autorregressive model is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &= \phi_0 + \sum_{i=1}^{lags} \phi_{i, t} y_{t - i} + \varepsilon_{t}  \quad \varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \phi_{i, t+1} &= \phi_{i, t} + \eta{i, t}  \quad \eta{i, t} \sim \mathcal{N}(0, \sigma^2_{i, \eta})\\
    \end{aligned}
\end{gather*}
```

!!!! note System matrices
    When building the system matrices we get rid of the first `lags` observations 
    in order to build all system matrices respecting the correspondent timestamps

!!! warning Forecasting
    The dynamic autorregressive model does not have the [`forecast`](@ref) method implemented yet. 
    If you wish to perform forecasts with this model please open an issue.

!!! warning Missing values
    The dynamic autorregressive model currently does not support missing values (`NaN` observations.)

# Example
```jldoctest
julia> model = DAR(randn(100), 2)
DAR
```
"""
mutable struct DAR <: StateSpaceModel
    lags::Int
    first_observations::Vector{<:Real}
    hyperparameters::HyperParameters
    system::LinearUnivariateTimeVariant
    results::Results

    function DAR(y::Vector{Fl}, lags::Int) where Fl

        assert_zero_missing_values(y)

        X = lagmat(y, lags)
        num_observations = size(X, 1)
        first_observations = y[1:lags]

        # Define system matrices
        Z = [X[t, :] for t in 1:num_observations]
        T = [Matrix{Fl}(I, lags, lags) for _ in 1:num_observations]
        R = [Matrix{Fl}(I, lags, lags) for _ in 1:num_observations]
        d = [zero(Fl) for _ in 1:num_observations]
        c = [zeros(lags) for _ in 1:num_observations]
        H = [one(Fl) for _ in 1:num_observations]
        Q = [Matrix{Fl}(I, lags, lags) for _ in 1:num_observations]

        system = LinearUnivariateTimeVariant{Fl}(y[lags+1:end], Z, T, R, d, c, H, Q)

        # Define hyperparameters names
        names = [["sigma2_ε", "intercept"];["sigma2_ar_$i" for i in 1:lags]]
        hyperparameters = HyperParameters{Fl}(names)

        return new(lags, first_observations, hyperparameters, system, Results{Fl}())
    end
end

# Obligatory methods
function default_filter(model::DAR)
    Fl = typeof_model_elements(model)
    a1 = zeros(Fl, num_states(model))
    # P1 is identity on sigma \eta and 0 otherwise.
    P1 = Matrix{Fl}(I, num_states(model), num_states(model))
    steadystate_tol = Fl(1e-5)
    return UnivariateKalmanFilter(a1, P1, 0, steadystate_tol)
end

function get_sigma_ar_name(model::DAR, i::Int)
    return model.hyperparameters.names[i + 2]
end

function initial_hyperparameters!(model::DAR)
    Fl = typeof_model_elements(model)
    observed_variance = variance_of_valid_observations(model.system.y)
    observed_mean = mean_of_valid_observations(model.system.y)
    initial_hyperparameters = Dict{String,Fl}(
        "sigma2_ε" => observed_variance, "intercept" => observed_mean
    )
    for i in axes(model.system.T[1], 2)
        initial_hyperparameters[get_sigma_ar_name(model, i)] = one(Fl)
    end
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::DAR)
    for i in axes(model.system.T[1], 2)
        constrain_variance!(model, get_sigma_ar_name(model, i))
    end
    constrain_variance!(model, "sigma2_ε")
    constrain_identity!(model, "intercept")
    return model
end

function unconstrain_hyperparameters!(model::DAR)
    for i in axes(model.system.T[1], 2)
        unconstrain_variance!(model, get_sigma_ar_name(model, i))
    end
    unconstrain_variance!(model, "sigma2_ε")
    unconstrain_identity!(model, "intercept")
    return model
end

function fill_model_system!(model::DAR)
    H = get_constrained_value(model, "sigma2_ε")
    fill_H_in_time(model, H)
    intercept = get_constrained_value(model, "intercept")
    for t in 1:length(model.system.Q)
        fill!(model.system.d, intercept)
        for i in axes(model.system.Q[1], 1)
            model.system.Q[t][i, i] = get_constrained_value(model, get_sigma_ar_name(model, i))
        end
    end
    return model
end

function fill_H_in_time(model::DAR, H::Fl) where Fl
    return fill_system_matrice_with_value_in_time(model.system.H, H)
end

function reinstantiate(model::DAR, y::Vector{Fl}) where Fl
    return DAR(y, model.lags)
end

# This model needs a custom made forecasting routine
function forecast(
    model::DAR, steps_ahead::Int; filter::KalmanFilter=default_filter(model)
)
    return error("Forecasting is currently nor implemented for the DAR model.")
end