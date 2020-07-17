export LocalLevel

@doc raw"""
The local level model is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t}  + \varepsilon_{t} \quad \varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \eta_{t} \quad \eta_{t} \sim \mathcal{N}(0, \sigma^2_{\eta})\\
    \end{aligned}
\end{gather*}
```

# Example
```jldoctest
julia> model = LocalLevel(rand(100))
A LocalLevel{Float64} model
```

See more on [Nile river annual flow](@ref)

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. pp. 9

"""
struct LocalLevel{Fl <: Real} <: StateSpaceModel
    hyperparameters::HyperParameters{Fl}
    system::LinearUnivariateTimeInvariant{Fl}

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

        return new{Fl}(hyperparameters, system)
    end
end

# Obligatory methods
function default_filter(::LocalLevel{Fl}) where Fl
    a1 = zero(Fl)
    P1 = Fl(1e6)
    steadystate_tol = Fl(1e-5)
    return ScalarKalmanFilter(a1, P1, 1, steadystate_tol)
end
function initial_hyperparameters!(model::LocalLevel{Fl}) where Fl
    initial_hyperparameters = Dict{String, Fl}(
        "sigma2_ε" => var(model.system.y),
        "sigma2_η" => var(model.system.y)
    )
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return
end
function constraint_hyperparameters!(model::LocalLevel{Fl}) where Fl
    constrain_variance(model, "sigma2_ε")
    constrain_variance(model, "sigma2_η")
    return
end
function unconstraint_hyperparameters!(model::LocalLevel{Fl}) where Fl
    unconstrain_variance(model, "sigma2_ε")
    unconstrain_variance(model, "sigma2_η")
    return
end
function update!(model::LocalLevel{Fl}) where Fl
    model.system.H = get_constrained_value(model, "sigma2_ε")
    model.system.Q[1] = get_constrained_value(model, "sigma2_η")
    return 
end
