@doc raw"""
    BasicStructuralExplanatory(y::Vector{Fl}, s::Int, X::Matrix{Fl}) where Fl

It is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \gamma_{t} + \beta_{t, i}X_{t, i} \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \nu_{t} + \xi_{t} \quad &\xi_{t} \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \nu_{t+1} &= \nu_{t} + \zeta_{t} \quad &\zeta_{t} \sim \mathcal{N}(0, \sigma^2_{\zeta})\\
        \gamma_{t+1} &= -\sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_{t} \quad & \omega_{t} \sim \mathcal{N}(0, \sigma^2_{\omega})\\
        \beta_{t+1} &= \beta_{t}
    \end{aligned}
\end{gather*}
```

# Example
```jldoctest
julia> model = BasicStructuralExplanatory(rand(100), 12, rand(100, 2))
BasicStructuralExplanatory
```

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press.
"""
mutable struct BasicStructuralExplanatory <: StateSpaceModel
    hyperparameters::HyperParameters
    system::LinearUnivariateTimeVariant
    seasonality::Int
    results::Results
    exogenous::Matrix

    function BasicStructuralExplanatory(y::Vector{Fl}, s::Int, X::Matrix{Fl}) where Fl

        @assert length(y) == size(X, 1)

        num_observations = size(X, 1)
        num_exogenous = size(X, 2)

        Z = [vcat([1; 0; 1; zeros(Fl, s - 2)], X[t, :]) for t in 1:num_observations]
        T = [[
            1 1 zeros(Fl, 1, s - 1) zeros(Fl, 1, num_exogenous)
            0 1 zeros(Fl, 1, s - 1) zeros(Fl, 1, num_exogenous)
            0 0 -ones(Fl, 1, s - 1) zeros(Fl, 1, num_exogenous)
            zeros(Fl, s - 2, 2) Matrix{Fl}(I, s - 2, s - 2) zeros(Fl, s - 2) zeros(Fl, s - 2, num_exogenous)
            zeros(Fl, num_exogenous, s + 1) Matrix{Fl}(I, num_exogenous, num_exogenous) 
        ] for _ in 1:num_observations]
        R = [[
            Matrix{Fl}(I, 3, 3)
            zeros(Fl, s - 2, 3)
            zeros(num_exogenous, 3)
        ] for _ in 1:num_observations]
        d = [zero(Fl) for _ in 1:num_observations]
        c = [zeros(Fl, size(T[1], 1)) for _ in 1:num_observations]
        H = [one(Fl) for _ in 1:num_observations]
        Q = [zeros(Fl, 3, 3) for _ in 1:num_observations]

        system = LinearUnivariateTimeVariant{Fl}(y, Z, T, R, d, c, H, Q)

        names = [["sigma2_ε", "sigma2_ξ", "sigma2_ζ", "sigma2_ω"]; ["β_$i" for i in 1:num_exogenous]]

        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, system, s, Results{Fl}(), X)
    end
end

function default_filter(model::BasicStructuralExplanatory)
    Fl = typeof_model_elements(model)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, num_states(model))
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    return UnivariateKalmanFilter(a1, P1, num_states(model), steadystate_tol)
end

function initial_hyperparameters!(model::BasicStructuralExplanatory)
    Fl = typeof_model_elements(model)
    initial_hyperparameters = Dict{String,Fl}(
        "sigma2_ε" => one(Fl),
        "sigma2_ξ" => one(Fl),
        "sigma2_ζ" => one(Fl),
        "sigma2_ω" => one(Fl),
    )
    initial_exogenous = model.exogenous \ model.system.y
    for i in axes(model.exogenous, 2)
        initial_hyperparameters[get_beta_name(model, i)] = initial_exogenous[i]
    end
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function get_beta_name(model::BasicStructuralExplanatory, i::Int)
    return model.hyperparameters.names[i + 4]
end

function constrain_hyperparameters!(model::BasicStructuralExplanatory)
    for i in axes(model.exogenous, 2)
        constrain_identity!(model, get_beta_name(model, i))
    end
    constrain_variance!(model, "sigma2_ε")
    constrain_variance!(model, "sigma2_ξ")
    constrain_variance!(model, "sigma2_ζ")
    constrain_variance!(model, "sigma2_ω")
    return model
end

function unconstrain_hyperparameters!(model::BasicStructuralExplanatory)
    for i in axes(model.exogenous, 2)
        unconstrain_identity!(model, get_beta_name(model, i))
    end
    unconstrain_variance!(model, "sigma2_ε")
    unconstrain_variance!(model, "sigma2_ξ")
    unconstrain_variance!(model, "sigma2_ζ")
    unconstrain_variance!(model, "sigma2_ω")
    return model
end

function fill_model_system!(model::BasicStructuralExplanatory)
    H = get_constrained_value(model, "sigma2_ε")
    fill_H_in_time(model, H)
    for t in 1:length(model.system.Q)
        model.system.Q[t][1] = get_constrained_value(model, "sigma2_ξ")
        model.system.Q[t][5] = get_constrained_value(model, "sigma2_ζ")
        model.system.Q[t][end] = get_constrained_value(model, "sigma2_ω")
    end
    return model
end

function fill_H_in_time(model::BasicStructuralExplanatory, H::Fl) where Fl
    return fill_system_matrice_with_value_in_time(model.system.H, H)
end

function reinstantiate(model::BasicStructuralExplanatory, y::Vector{Fl}, X::Matrix{Fl}) where Fl
    return BasicStructuralExplanatory(y, model.seasonality, X)
end

has_exogenous(::BasicStructuralExplanatory) = true

# BasicStructuralExplanatory requires a custom simulation

function simulate_scenarios(
    model::BasicStructuralExplanatory,
    steps_ahead::Int,
    n_scenarios::Int,
    new_exogenous::Matrix{Fl};
    filter::KalmanFilter=default_filter(model),
) where Fl
    @assert steps_ahead == size(new_exogenous, 1) "new_exogenous must have the same dimension as steps_ahead"
    # Query the type of model elements
    fo = kalman_filter(model)
    last_state = fo.a[end]
    num_series = size(model.system.y, 2)

    scenarios = Array{Fl,3}(undef, steps_ahead, num_series, n_scenarios)
    for s in 1:n_scenarios
        scenarios[:, :, s] = simulate(model, last_state, steps_ahead, new_exogenous)
    end
    return scenarios
end

function simulate_scenarios(
    model::BasicStructuralExplanatory,
    steps_ahead::Int,
    n_scenarios::Int,
    new_exogenous::Array{Fl, 3};
    filter::KalmanFilter=default_filter(model),
) where Fl
    @assert steps_ahead == size(new_exogenous, 1) "new_exogenous must have the same dimension of steps_ahead"
    @assert n_scenarios == size(new_exogenous, 3) "new_exogenous must have the same number of scenarios of n_scenarios"
    # Query the type of model elements
    fo = kalman_filter(model)
    last_state = fo.a[end]
    num_series = size(model.system.y, 2)

    scenarios = Array{Fl,3}(undef, steps_ahead, num_series, n_scenarios)
    for s in 1:n_scenarios
        scenarios[:, :, s] = simulate(model, last_state, steps_ahead, new_exogenous[:, :, s])
    end
    return scenarios
end

function simulate(
    model::BasicStructuralExplanatory,
    initial_state::Vector{Fl},
    n::Int,
    new_exogenous::Matrix{Fl};
    return_simulated_states::Bool=false,
) where Fl
    sys = model.system
    m = size(sys.T[1], 1)
    y = Vector{Fl}(undef, n)
    alpha = Matrix{Fl}(undef, n + 1, m)
    # Sampling errors
    chol_H = sqrt(sys.H[1])
    chol_Q = cholesky_decomposition(sys.Q[1])
    standard_ε = randn(n)
    standard_η = randn(n + 1, size(sys.Q[1], 1))

    num_exogenous = size(model.exogenous, 2)
    @assert num_exogenous == size(new_exogenous, 2) "You must have the same number of exogenous variables of the model."

    # The first state of the simulation is the update of a_0
    alpha[1, :] .= initial_state
    sys.Z[1][end-num_exogenous+1:end] .= new_exogenous[1, :]
    y[1] = dot(sys.Z[1], initial_state) + sys.d[1] + chol_H * standard_ε[1]
    alpha[2, :] = sys.T[1] * initial_state + sys.c[1] + sys.R[1] * chol_Q.L * standard_η[1, :]
    # Simulate scenarios
    for t in 2:n
        sys.Z[t][end-num_exogenous+1:end] .= new_exogenous[t, :]
        y[t] = dot(sys.Z[t], alpha[t, :]) + sys.d[t] + chol_H * standard_ε[t]
        alpha[t + 1, :] = sys.T[t] * alpha[t, :] + sys.c[t] + sys.R[t] * chol_Q.L * standard_η[t, :]
    end

    if return_simulated_states
        return y, alpha[1:n, :]
    end
    return y
end