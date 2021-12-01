@doc raw"""
    LocalLevelCycle(y::Vector{Fl}) where Fl

The local level model with a cycle component is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + c_{t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \eta_{t} \quad &\eta_{t} \sim \mathcal{N}(0, \sigma^2_{\eta})\\
        c_{t+1} &= c_{t} \cos(\lambda_c) + c_{t}^{*} \sin(\lambda_c)\ \quad & \tilde\omega_{t} \sim \mathcal{N}(0, \sigma^2_{\tilde\omega})\\
        c_{t+1}^{*} &= -c_{t} \sin(\lambda_c) + c_{t}^{*} \sin(\lambda_c) \quad &\tilde\omega^*_{t} \sim \mathcal{N}(0, \sigma^2_{\tilde\omega})\\
    \end{aligned}
\end{gather*}
```

# Example
```jldoctest
julia> model = LocalLevelCycle(rand(100))
LocalLevelCycle
```

See more on TODO RJ_TEMPERATURE

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. pp. 48
"""
mutable struct LocalLevelCycle <: StateSpaceModel
    hyperparameters::HyperParameters
    system::LinearUnivariateTimeInvariant
    results::Results

    function LocalLevelCycle(y::Vector{Fl}) where Fl

        # Define system matrices
        Z = Fl[1, 1, 0]
        T = Fl[
            1 0 0
            0 0 0
            0 0 0
        ]
        R = Matrix{Fl}(I, 3, 3)
        d = zero(Fl)
        c = zeros(Fl, 3)
        H = one(Fl)
        Q = Matrix{Fl}(I, 3, 3)

        system = LinearUnivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        # Define hyperparameters names
        names = ["sigma2_ε", "sigma2_η", "sigma2_ω", "λ_c"]
        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, system, Results{Fl}())
    end
end

# Obligatory methods
function default_filter(model::LocalLevelCycle)
    Fl = typeof_model_elements(model)
    a1 = zeros(Fl, num_states(model))
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    steadystate_tol = Fl(1e-5)
    return UnivariateKalmanFilter(a1, P1, num_states(model), steadystate_tol)
end

function initial_hyperparameters!(model::LocalLevelCycle)
    Fl = typeof_model_elements(model)
    observed_variance = variance_of_valid_observations(model.system.y)
    initial_hyperparameters = Dict{String,Fl}(
        "sigma2_ε" => observed_variance,
        "sigma2_η" => observed_variance,
        "sigma2_ω" => one(Fl),
        # Durbin and Koopman (2012) comment possible values
        # in their book pp. 48
        "λ_c" => Fl(2 * pi / 12),
    )
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::LocalLevelCycle)
    Fl = typeof_model_elements(model)
    constrain_variance!(model, "sigma2_ε")
    constrain_variance!(model, "sigma2_η")
    constrain_variance!(model, "sigma2_ω")
    # Durbin and Koopman (2012) comment possible values in their book pp. 48
    constrain_box!(model, "λ_c", Fl(2 * pi / 100), Fl(2 * pi / 1.5))
    return model
end

function unconstrain_hyperparameters!(model::LocalLevelCycle)
    Fl = typeof_model_elements(model)
    unconstrain_variance!(model, "sigma2_ε")
    unconstrain_variance!(model, "sigma2_η")
    unconstrain_variance!(model, "sigma2_ω")
    # Durbin and Koopman (2012) comment possible values in their book pp. 48
    unconstrain_box!(model, "λ_c", Fl(2 * pi / 100), Fl(2 * pi / 1.5))
    return model
end

function fill_model_system!(model::LocalLevelCycle)
    model.system.H = get_constrained_value(model, "sigma2_ε")
    model.system.Q[1] = get_constrained_value(model, "sigma2_η")
    model.system.Q[5] = get_constrained_value(model, "sigma2_ω")
    model.system.Q[9] = get_constrained_value(model, "sigma2_ω")
    model.system.T[5] = cos(get_constrained_value(model, "λ_c"))
    model.system.T[6] = -sin(get_constrained_value(model, "λ_c"))
    model.system.T[8] = sin(get_constrained_value(model, "λ_c"))
    model.system.T[9] = cos(get_constrained_value(model, "λ_c"))
    return model
end

function reinstantiate(::LocalLevelCycle, y::Vector{Fl}) where Fl
    return LocalLevelCycle(y)
end

has_exogenous(::LocalLevelCycle) = false
