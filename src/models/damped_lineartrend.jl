"""
\\phi is between 0.8 and 1
"""
mutable struct DampedLinearTrend{Fl} <: StateSpaceModel
    hyperparameters::HyperParameters{Fl}
    system::LinearUnivariateTimeInvariant{Fl}

    function DampedLinearTrend(y::Vector{Fl}) where Fl

        Z = Fl.([1.0; 0.0])
        T = Fl.([1 1; 0 1])
        R = Fl.([1 0; 0 1])
        d = zero(Fl)
        c = zeros(Fl, 2)
        H = one(Fl)
        Q = Fl.([1 0; 0 1])

        system = LinearUnivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        names = ["ϕ", "sigma2_ε", "sigma2_η", "sigma2_β"]
        hyperparameters = HyperParameters{Fl}(names)

        return new{Fl}(hyperparameters, system)
    end
end

function default_filter(::DampedLinearTrend{Fl}) where Fl
    steadystate_tol = 1e-5
    a1 = zeros(Fl, 2)
    P1 = 1e6 .* Matrix{Fl}(I, 2, 2)
    return UnivariateKalmanFilter(a1, P1, 0, steadystate_tol)
end
function initial_hyperparameters!(model::DampedLinearTrend{Fl}) where Fl
    initial_hyperparameters = Dict{String, Fl}(
        "ϕ"        => Fl(5),
        "sigma2_ε" => var(model.system.y),
        "sigma2_η" => var(model.system.y),
        "sigma2_β" => one(Fl)
    )
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return
end
function constraint_hyperparameters!(model::DampedLinearTrend{Fl}) where {Fl}
    constrain_box(model, "ϕ", Fl(0.8), Fl(1.0))
    update_constrained_value!(model, "sigma2_ε", get_unconstrained_value(model, "sigma2_ε")^2)
    update_constrained_value!(model, "sigma2_η", get_unconstrained_value(model, "sigma2_η")^2)
    update_constrained_value!(model, "sigma2_β", get_unconstrained_value(model, "sigma2_β")^2)
    return
end
function unconstraint_hyperparameters!(model::DampedLinearTrend{Fl}) where Fl
    unconstrain_box(model, "ϕ", Fl(0.8), Fl(1.0))
    update_unconstrained_value!(model, "sigma2_ε", sqrt(get_constrained_value(model, "sigma2_ε")))
    update_unconstrained_value!(model, "sigma2_η", sqrt(get_constrained_value(model, "sigma2_η")))
    update_unconstrained_value!(model, "sigma2_β", sqrt(get_constrained_value(model, "sigma2_β")))
end
function update!(model::DampedLinearTrend{Fl}) where Fl
    model.system.T[end] = get_constrained_value(model, "ϕ")
    model.system.H = get_constrained_value(model, "sigma2_ε")
    model.system.Q[1] = get_constrained_value(model, "sigma2_η")
    model.system.Q[end] = get_constrained_value(model, "sigma2_β")
    return 
end