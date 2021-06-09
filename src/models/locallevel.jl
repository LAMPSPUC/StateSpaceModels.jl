@doc raw"""
    LocalLevel(y::Vector{Fl}) where Fl

The local level model is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \varepsilon_{t} \quad \varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \eta_{t} \quad \eta_{t} \sim \mathcal{N}(0, \sigma^2_{\eta})\\
    \end{aligned}
\end{gather*}
```

# Example
```jldoctest
julia> model = LocalLevel(rand(100))
LocalLevel
```

See more on [Nile river annual flow](@ref)

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. pp. 9
"""
mutable struct LocalLevel <: StateSpaceModel
    hyperparameters::HyperParameters
    system::LinearUnivariateTimeInvariant
    results::Results

    function LocalLevel(y::Vector{Fl}) where Fl

        # Define system matrices
        Z = ones(Fl, 1)
        T = ones(Fl, 1, 1)
        R = ones(Fl, 1, 1)
        d = zero(Fl)
        c = zeros(Fl, 1)
        H = one(Fl)
        Q = ones(Fl, 1, 1)

        system = LinearUnivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        # Define hyperparameters names
        names = ["sigma2_ε", "sigma2_η"]
        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, system, Results{Fl}())
    end
end

# Obligatory methods
function default_filter(model::LocalLevel)
    Fl = typeof_model_elements(model)
    a1 = zero(Fl)
    P1 = Fl(1e6)
    steadystate_tol = Fl(1e-5)
    return ScalarKalmanFilter(a1, P1, 1, steadystate_tol)
end

function initial_hyperparameters!(model::LocalLevel)
    Fl = typeof_model_elements(model)
    observed_variance = var(model.system.y[findall(!isnan, model.system.y)])
    initial_hyperparameters = Dict{String,Fl}(
        "sigma2_ε" => observed_variance, "sigma2_η" => observed_variance
    )
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::LocalLevel)
    constrain_variance!(model, "sigma2_ε")
    constrain_variance!(model, "sigma2_η")
    return model
end

function unconstrain_hyperparameters!(model::LocalLevel)
    unconstrain_variance!(model, "sigma2_ε")
    unconstrain_variance!(model, "sigma2_η")
    return model
end

function fill_model_system!(model::LocalLevel)
    model.system.H = get_constrained_value(model, "sigma2_ε")
    model.system.Q[1] = get_constrained_value(model, "sigma2_η")
    return model
end

function reinstantiate(::LocalLevel, y::Vector{Fl}) where Fl
    return LocalLevel(y)
end

has_exogenous(::LocalLevel) = false
