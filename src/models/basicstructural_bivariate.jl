@doc raw"""
A hand crafted bivariate basic structural state-space model consists of a trend (represented with a 
local level) and a seasonal component. It is defined by:
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \gamma_{t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \xi_{t} \quad &\xi_{t} \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \gamma_{t+1} &= -\sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_{t} \quad & \omega_{t} \sim \mathcal{N}(0, \sigma^2_{\omega})\\
    \end{aligned}
\end{gather*}
```
"""
mutable struct BivariateBasicStructural <: StateSpaceModel
    hyperparameters::HyperParameters
    system::LinearMultivariateTimeInvariant
    seasonality::Int
    results::Results

    function BivariateBasicStructural(y::Matrix{Fl}, s::Int) where Fl
        @assert size(y, 2) == 2
        p = 2
        Z = kron(Matrix{Fl}(I, p, p), [1 1 zeros(1, s - 2)])
        T = kron(Matrix{Fl}(I, p, p),
                [1 zeros(Fl, 1, s - 1);
                0 -ones(Fl, 1, s - 1);
                zeros(Fl, s - 2, 1) Matrix{Fl}(I, s - 2, s - 2) zeros(Fl, s - 2)])
                
        R = kron(Matrix{Fl}(I, p, p),
                [Matrix{Fl}(I, 2, 2);
                zeros(Fl, s - 2, 2)])

        d = zeros(Fl, 2)
        c = zeros(Fl, 2 * s)
        H = kron(Matrix{Fl}(I, p, p), one(Fl))
        Q_fixed = Matrix{Fl}(I, p, p) * 1e-5 # fixed small variaance
        Q = kron(Matrix{Fl}(I, p, p), zeros(Fl, 2, 2))
        Q[end-1:end, end-1:end] = Q_fixed # Fix small variance for seaasonal component

        system = LinearMultivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        names = ["sigma2_ε1", "sigma_ε1sigma_ε2", "sigma2_ε2",
                 "sigma2_ξ1", "sigma_ξ1sigma_ξ2", "sigma2_ξ2"]
        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, system, s, Results{Fl}())
    end
end

function default_filter(model::BivariateBasicStructural)
    Fl = typeof_model_elements(model)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, num_states(model))
    P1 = 1e6 .* Matrix{Fl}(I, num_states(model), num_states(model))
    return MultivariateKalmanFilter(2, a1, P1, num_states(model), steadystate_tol)
end
function initial_hyperparameters!(model::BivariateBasicStructural)
    Fl = typeof_model_elements(model)
    initial_hyperparameters = Dict{String, Fl}(
        "sigma2_ε1" => one(Fl),
        "sigma_ε1sigma_ε2" => zero(Fl),
        "sigma2_ε2" => one(Fl),
        "sigma2_ξ1" => one(Fl),
        "sigma_ξ1sigma_ξ2" => zero(Fl),
        "sigma2_ξ2" => one(Fl),
    )
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return
end
function constrain_hyperparameters!(model::BivariateBasicStructural)
    Fl = typeof_model_elements(model)
    # Constrain H via H*H'
    sigma2_ε1        = get_unconstrained_value(model, "sigma2_ε1")
    sigma_ε1sigma_ε2 = get_unconstrained_value(model, "sigma_ε1sigma_ε2")
    sigma2_ε2        = get_unconstrained_value(model, "sigma2_ε2")
    H_unconstrained = [sigma2_ε1        zero(Fl)
                       sigma_ε1sigma_ε2 sigma2_ε2]
    H_constrained = H_unconstrained * H_unconstrained'
    update_constrained_value!(model, "sigma2_ε1", H_constrained[1, 1])
    update_constrained_value!(model, "sigma_ε1sigma_ε2", H_constrained[2, 1])
    update_constrained_value!(model, "sigma2_ε2", H_constrained[2, 2])
    # Constrain Q via Q*Q'
    sigma2_ξ1        = get_unconstrained_value(model, "sigma2_ξ1")
    sigma_ξ1sigma_ξ2 = get_unconstrained_value(model, "sigma_ξ1sigma_ξ2")
    sigma2_ξ2        = get_unconstrained_value(model, "sigma2_ξ2")
    Q_unconstrained = [sigma2_ξ1        zero(Fl)
                       sigma_ξ1sigma_ξ2 sigma2_ξ2]
    Q_constrained = Q_unconstrained * Q_unconstrained'
    update_constrained_value!(model, "sigma2_ξ1", Q_constrained[1, 1])
    update_constrained_value!(model, "sigma_ξ1sigma_ξ2", Q_constrained[2, 1])
    update_constrained_value!(model, "sigma2_ξ2", Q_constrained[2, 2])
    return
end
function unconstrain_hyperparameters!(model::BivariateBasicStructural)
    # Unconstrain H via cholesky
    sigma2_ε1        = get_constrained_value(model, "sigma2_ε1")
    sigma_ε1sigma_ε2 = get_constrained_value(model, "sigma_ε1sigma_ε2")
    sigma2_ε2        = get_constrained_value(model, "sigma2_ε2")
    H_constrained = [sigma2_ε1        sigma_ε1sigma_ε2
                     sigma_ε1sigma_ε2 sigma2_ε2]
    H_unconstrained = cholesky(H_constrained).L
    update_unconstrained_value!(model, "sigma2_ε1", H_unconstrained[1, 1])
    update_unconstrained_value!(model, "sigma_ε1sigma_ε2", H_unconstrained[2, 1])
    update_unconstrained_value!(model, "sigma2_ε2", H_unconstrained[2, 2])
    # Unconstrain Q via cholesky
    sigma2_ξ1        = get_constrained_value(model, "sigma2_ξ1")
    sigma_ξ1sigma_ξ2 = get_constrained_value(model, "sigma_ξ1sigma_ξ2")
    sigma2_ξ2        = get_constrained_value(model, "sigma2_ξ2")
    Q_constrained = [sigma2_ξ1        sigma_ξ1sigma_ξ2
                     sigma_ξ1sigma_ξ2 sigma2_ξ2]
    Q_unconstrained = cholesky(Q_constrained).L
    update_unconstrained_value!(model, "sigma2_ξ1", Q_unconstrained[1, 1])
    update_unconstrained_value!(model, "sigma_ξ1sigma_ξ2", Q_unconstrained[2, 1])
    update_unconstrained_value!(model, "sigma2_ξ2", Q_unconstrained[2, 2])
    return
end
function fill_model_system!(model::BivariateBasicStructural)
    model.system.H[1, 1] = get_constrained_value(model, "sigma2_ε1")
    model.system.H[2, 1] = get_constrained_value(model, "sigma_ε1sigma_ε2")
    model.system.H[1, 2] = get_constrained_value(model, "sigma_ε1sigma_ε2")
    model.system.H[2, 2] = get_constrained_value(model, "sigma2_ε2")
    model.system.Q[1, 1] = get_constrained_value(model, "sigma2_ξ1")
    model.system.Q[2, 1] = get_constrained_value(model, "sigma_ξ1sigma_ξ2")
    model.system.Q[1, 2] = get_constrained_value(model, "sigma_ξ1sigma_ξ2")
    model.system.Q[2, 2] = get_constrained_value(model, "sigma2_ξ2")
    return
end
function reinstantiate(model::BivariateBasicStructural, y::Matrix{Fl}) where Fl
    return BivariateBasicStructural(y, model.seasonality)
end
has_exogenous(::BivariateBasicStructural) = false