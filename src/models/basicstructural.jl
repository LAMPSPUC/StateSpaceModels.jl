@doc raw"""
    BasicStructural(y::Vector{Fl}, s::Int) where Fl

The basic structural state-space model consists of a trend (level + slope) and a seasonal
component. It is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \gamma_{t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \nu_{t} + \xi_{t} \quad &\xi_{t} \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \nu_{t+1} &= \nu_{t} + \zeta_{t} \quad &\zeta_{t} \sim \mathcal{N}(0, \sigma^2_{\zeta})\\
        \gamma_{t+1} &= -\sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_{t} \quad & \omega_{t} \sim \mathcal{N}(0, \sigma^2_{\omega})\\
    \end{aligned}
\end{gather*}
```

# Example
```jldoctest
julia> model = BasicStructural(rand(100), 12)
BasicStructural
```

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press.
"""
mutable struct BasicStructural <: StateSpaceModel
    hyperparameters::HyperParameters
    system::LinearUnivariateTimeInvariant
    seasonality::Int
    results::Results

    function BasicStructural(y::Vector{Fl}, s::Int) where Fl
        Z = [1; 0; 1; zeros(Fl, s - 2)]
        T = [
            1 1 zeros(Fl, 1, s - 1)
            0 1 zeros(Fl, 1, s - 1)
            0 0 -ones(Fl, 1, s - 1)
            zeros(Fl, s - 2, 2) Matrix{Fl}(I, s - 2, s - 2) zeros(Fl, s - 2)
        ]
        R = [
            Matrix{Fl}(I, 3, 3)
            zeros(Fl, s - 2, 3)
        ]
        d = zero(Fl)
        c = zeros(Fl, s + 1)
        H = one(Fl)
        Q = zeros(Fl, 3, 3)

        system = LinearUnivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        names = ["sigma2_ε", "sigma2_ξ", "sigma2_ζ", "sigma2_ω"]
        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, system, s, Results{Fl}())
    end
end

function default_filter(model::BasicStructural)
    Fl = typeof_model_elements(model)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, num_states(model))
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    return UnivariateKalmanFilter(a1, P1, num_states(model), steadystate_tol)
end

function initial_hyperparameters!(model::BasicStructural)
    Fl = typeof_model_elements(model)
    initial_hyperparameters = Dict{String,Fl}(
        "sigma2_ε" => one(Fl),
        "sigma2_ξ" => one(Fl),
        "sigma2_ζ" => one(Fl),
        "sigma2_ω" => one(Fl),
    )
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::BasicStructural)
    constrain_variance!(model, "sigma2_ε")
    constrain_variance!(model, "sigma2_ξ")
    constrain_variance!(model, "sigma2_ζ")
    constrain_variance!(model, "sigma2_ω")
    return model
end

function unconstrain_hyperparameters!(model::BasicStructural)
    unconstrain_variance!(model, "sigma2_ε")
    unconstrain_variance!(model, "sigma2_ξ")
    unconstrain_variance!(model, "sigma2_ζ")
    unconstrain_variance!(model, "sigma2_ω")
    return model
end

function fill_model_system!(model::BasicStructural)
    model.system.H = get_constrained_value(model, "sigma2_ε")
    model.system.Q[1] = get_constrained_value(model, "sigma2_ξ")
    model.system.Q[5] = get_constrained_value(model, "sigma2_ζ")
    model.system.Q[end] = get_constrained_value(model, "sigma2_ω")
    return model
end

function reinstantiate(model::BasicStructural, y::Vector{Fl}) where Fl
    return BasicStructural(y, model.seasonality)
end

has_exogenous(::BasicStructural) = false
