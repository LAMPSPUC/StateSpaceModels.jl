@doc raw"""
    LocalLinearTrend(y::Vector{Fl}) where Fl

The linear trend model is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \gamma_{t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \nu_{t} + \xi_{t} \quad &\xi_{t} \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \nu_{t+1} &= \nu_{t} + \zeta_{t} \quad &\zeta_{t} \sim \mathcal{N}(0, \sigma^2_{\zeta})\\
    \end{aligned}
\end{gather*}
```

# Example
```jldoctest
julia> model = LocalLinearTrend(rand(100))
LocalLinearTrend
```

See more on [Finland road traffic fatalities](@ref)

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods:
    Second Edition." Oxford University Press. pp. 44
"""
mutable struct LocalLinearTrend <: StateSpaceModel
    hyperparameters::HyperParameters
    system::LinearUnivariateTimeInvariant
    results::Results

    function LocalLinearTrend(y::Vector{Fl}) where Fl
        Z = Fl.([1.0; 0.0])
        T = Fl.([1 1; 0 1])
        R = Fl.([1 0; 0 1])
        d = zero(Fl)
        c = zeros(Fl, 2)
        H = one(Fl)
        Q = Fl.([1 0; 0 1])

        system = LinearUnivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        names = ["sigma2_ε", "sigma2_ξ", "sigma2_ζ"]
        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, system, Results{Fl}())
    end
end

function default_filter(model::LocalLinearTrend)
    Fl = typeof_model_elements(model)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, num_states(model))
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    return UnivariateKalmanFilter(a1, P1, num_states(model), steadystate_tol)
end

function initial_hyperparameters!(model::LocalLinearTrend)
    Fl = typeof_model_elements(model)
    observed_variance = variance_of_valid_observations(model.system.y)
    initial_hyperparameters = Dict{String,Fl}(
        "sigma2_ε" => observed_variance,
        "sigma2_ξ" => observed_variance,
        "sigma2_ζ" => one(Fl),
    )
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::LocalLinearTrend)
    constrain_variance!(model, "sigma2_ε")
    constrain_variance!(model, "sigma2_ξ")
    constrain_variance!(model, "sigma2_ζ")
    return model
end

function unconstrain_hyperparameters!(model::LocalLinearTrend)
    unconstrain_variance!(model, "sigma2_ε")
    unconstrain_variance!(model, "sigma2_ξ")
    unconstrain_variance!(model, "sigma2_ζ")
    return model
end

function fill_model_system!(model::LocalLinearTrend)
    model.system.H = get_constrained_value(model, "sigma2_ε")
    model.system.Q[1] = get_constrained_value(model, "sigma2_ξ")
    model.system.Q[end] = get_constrained_value(model, "sigma2_ζ")
    return model
end

function reinstantiate(::LocalLinearTrend, y::Vector{Fl}) where Fl
    return LocalLinearTrend(y)
end

has_exogenous(::LocalLinearTrend) = false
