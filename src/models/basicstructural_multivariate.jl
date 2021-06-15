@doc raw"""
    MultivariateBasicStructural(y::Matrix{Fl}, s::Int) where Fl

An implementation of a non-homogeneous seemingly unrelated time series equations for basic structural
state-space model consists of trend (local linear trend) and seasonal components.
It is defined by:
```math
\begin{gather*}
    \begin{aligned}
    y_{t} &=  \mu_{t} + \gamma_{t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \Sigma_{\varepsilon})\\
    \mu_{t+1} &= \mu_{t} + \nu_{t} + \xi_{t} \quad &\xi_{t} \sim \mathcal{N}(0, \Sigma_{\xi})\\
    \nu_{t+1} &= \nu_{t} + \zeta_{t} \quad &\zeta_{t} \sim \mathcal{N}(0, \Sigma_{\zeta})\\
    \gamma_{t+1} &= -\sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_{t} \quad & \omega_{t} \sim \mathcal{N}(0, \Sigma_{\omega})\\
    \end{aligned}
\end{gather*}
```

# Example
```jldoctest
julia> model = MultivariateBasicStructural(rand(100, 2), 12)
MultivariateBasicStructural
```

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press.
"""
mutable struct MultivariateBasicStructural <: StateSpaceModel
    hyperparameters::HyperParameters
    system::LinearMultivariateTimeInvariant
    seasonality::Int
    results::Results

    function MultivariateBasicStructural(y::Matrix{Fl}, s::Int) where Fl
        p = size(y, 2)
        Z = kron(Matrix{Fl}(I, p, p), [1 0 1 zeros(Fl, 1, s - 2)])
        T = kron(
            Matrix{Fl}(I, p, p),
            [
                1 1 zeros(Fl, 1, s - 1)
                0 1 zeros(Fl, 1, s - 1)
                0 0 -ones(Fl, 1, s - 1)
                zeros(Fl, s - 2, 2) Matrix{Fl}(I, s - 2, s - 2) zeros(Fl, s - 2)
            ],
        )

        R = kron(
            Matrix{Fl}(I, p, p),
            [
                Matrix{Fl}(I, 3, 3)
                zeros(Fl, s - 2, 3)
            ],
        )

        d = zeros(Fl, p)
        c = zeros(Fl, p * (s + 1))
        H = kron(one(Fl), Matrix{Fl}(I, p, p))
        Q = kron(ones(Fl, p, p), Matrix{Fl}(I, 3, 3))

        system = LinearMultivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        names = handle_multivariate_basicstructural_names(p)
        hyperparameters = HyperParameters{Fl}(names)
        return new(hyperparameters, system, s, Results{Fl}())
    end
end

function handle_multivariate_basicstructural_names(p::Int)
    # generate \varepsilon  names
    greek_letters_for_states = ["ε", "ξ", "ζ", "ω"]
    names = String[]
    # Walk the lower triangle positions
    for letter in greek_letters_for_states, i in 1:p, j in 1:i
        if i == j
            push!(names, "sigma2_$(letter)$(i)")
        else
            push!(names, "sigma_$(letter)$(i)sigma_$(letter)$(j)")
        end
    end
    return names
end

function default_filter(model::MultivariateBasicStructural)
    Fl = typeof_model_elements(model)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, num_states(model))
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    return MultivariateKalmanFilter(
        size(model.system.y, 2), a1, P1, num_states(model), steadystate_tol
    )
end

function initial_hyperparameters!(model::MultivariateBasicStructural)
    Fl = typeof_model_elements(model)
    initial_hyperparameters = Dict{String,Fl}()
    for variable in get_names(model)
        if occursin("sigma2", variable)
            initial_hyperparameters[variable] = one(Fl)
        else
            initial_hyperparameters[variable] = zero(Fl)
        end
    end
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return nothing
end

function constrain_hyperparameters!(model::MultivariateBasicStructural)
    Fl = typeof_model_elements(model)
    p = size(model.system.y, 2)
    # H
    H_unconstrained = zeros(Fl, p, p)
    current_name = 1
    for i in 1:p, j in 1:i
        H_unconstrained[i, j] = get_unconstrained_value(
            model, get_names(model)[current_name]
        )
        current_name += 1
    end
    H_constrained = H_unconstrained * H_unconstrained'
    # Q
    Q_unconstrained = zeros(Fl, 3p, 3p)
    sigma2_ξ = filter(x -> occursin("sigma2_ξ", x), get_names(model))
    sigma2_ξ_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, 0)))
        if mod1(i, 3) == 1
            Q_unconstrained[pos] = get_unconstrained_value(
                model, sigma2_ξ[sigma2_ξ_counter]
            )
            sigma2_ξ_counter += 1
        end
    end
    sigma2_ζ = filter(x -> occursin("sigma2_ζ", x), get_names(model))
    sigma2_ζ_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, 0)))
        if mod1(i, 3) == 2
            Q_unconstrained[pos] = get_unconstrained_value(
                model, sigma2_ζ[sigma2_ζ_counter]
            )
            sigma2_ζ_counter += 1
        end
    end
    sigma2_ω = filter(x -> occursin("sigma2_ω", x), get_names(model))
    sigma2_ω_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, 0)))
        if mod1(i, 3) == 3
            Q_unconstrained[pos] = get_unconstrained_value(
                model, sigma2_ω[sigma2_ω_counter]
            )
            sigma2_ω_counter += 1
        end
    end
    sigma_ξxsigma_ξx = filter(x -> occursin("sigma_ξ", x), get_names(model))
    sigma_ξxsigma_ξx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, -3)))
        if mod1(i, 3) == 1
            Q_unconstrained[pos] = get_unconstrained_value(
                model, sigma_ξxsigma_ξx[sigma_ξxsigma_ξx_counter]
            )
            sigma_ξxsigma_ξx_counter += 1
        end
    end
    sigma_ζxsigma_ζx = filter(x -> occursin("sigma_ζ", x), get_names(model))
    sigma_ζxsigma_ζx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, -3)))
        if mod1(i, 3) == 2
            Q_unconstrained[pos] = get_unconstrained_value(
                model, sigma_ζxsigma_ζx[sigma_ζxsigma_ζx_counter]
            )
            sigma_ζxsigma_ζx_counter += 1
        end
    end
    sigma_ωxsigma_ωx = filter(x -> occursin("sigma_ω", x), get_names(model))
    sigma_ωxsigma_ωx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, -3)))
        if mod1(i, 3) == 3
            Q_unconstrained[pos] = get_unconstrained_value(
                model, sigma_ωxsigma_ωx[sigma_ωxsigma_ωx_counter]
            )
            sigma_ωxsigma_ωx_counter += 1
        end
    end
    Q_constrained = Q_unconstrained * Q_unconstrained'

    # Updates
    # H
    current_name = 1
    for i in 1:p, j in 1:i
        update_constrained_value!(
            model, get_names(model)[current_name], H_constrained[i, j]
        )
        current_name += 1
    end
    # Q
    sigma2_ξ_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, 0)))
        if mod1(i, 3) == 1
            update_constrained_value!(model, sigma2_ξ[sigma2_ξ_counter], Q_constrained[pos])
            sigma2_ξ_counter += 1
        end
    end
    sigma2_ζ_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, 0)))
        if mod1(i, 3) == 2
            update_constrained_value!(model, sigma2_ζ[sigma2_ζ_counter], Q_constrained[pos])
            sigma2_ζ_counter += 1
        end
    end
    sigma2_ω_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, 0)))
        if mod1(i, 3) == 3
            update_constrained_value!(model, sigma2_ω[sigma2_ω_counter], Q_constrained[pos])
            sigma2_ω_counter += 1
        end
    end
    sigma_ξxsigma_ξx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, -3)))
        if mod1(i, 3) == 1
            update_constrained_value!(
                model, sigma_ξxsigma_ξx[sigma_ξxsigma_ξx_counter], Q_constrained[pos]
            )
            sigma_ξxsigma_ξx_counter += 1
        end
    end
    sigma_ζxsigma_ζx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, -3)))
        if mod1(i, 3) == 2
            update_constrained_value!(
                model, sigma_ζxsigma_ζx[sigma_ζxsigma_ζx_counter], Q_constrained[pos]
            )
            sigma_ζxsigma_ζx_counter += 1
        end
    end
    sigma_ωxsigma_ωx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, -3)))
        if mod1(i, 3) == 3
            update_constrained_value!(
                model, sigma_ωxsigma_ωx[sigma_ωxsigma_ωx_counter], Q_constrained[pos]
            )
            sigma_ωxsigma_ωx_counter += 1
        end
    end
    return model
end

function unconstrain_hyperparameters!(model::MultivariateBasicStructural)
    Fl = typeof_model_elements(model)
    p = size(model.system.y, 2)
    # H
    H_constrained = zeros(Fl, p, p)
    current_name = 1
    for i in 1:p, j in 1:i
        H_constrained[i, j] = get_constrained_value(model, get_names(model)[current_name])
        current_name += 1
    end
    H_unconstrained = cholesky(Symmetric(H_constrained, :L)).L
    # Q
    Q_constrained = zeros(Fl, 3p, 3p)
    sigma2_ξ = filter(x -> occursin("sigma2_ξ", x), get_names(model))
    sigma2_ξ_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, 0)))
        if mod1(i, 3) == 1
            Q_constrained[pos] = get_constrained_value(model, sigma2_ξ[sigma2_ξ_counter])
            sigma2_ξ_counter += 1
        end
    end
    sigma2_ζ = filter(x -> occursin("sigma2_ζ", x), get_names(model))
    sigma2_ζ_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, 0)))
        if mod1(i, 3) == 2
            Q_constrained[pos] = get_constrained_value(model, sigma2_ζ[sigma2_ζ_counter])
            sigma2_ζ_counter += 1
        end
    end
    sigma2_ω = filter(x -> occursin("sigma2_ω", x), get_names(model))
    sigma2_ω_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, 0)))
        if mod1(i, 3) == 3
            Q_constrained[pos] = get_constrained_value(model, sigma2_ω[sigma2_ω_counter])
            sigma2_ω_counter += 1
        end
    end
    sigma_ξxsigma_ξx = filter(x -> occursin("sigma_ξ", x), get_names(model))
    sigma_ξxsigma_ξx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, -3)))
        if mod1(i, 3) == 1
            Q_constrained[pos] = get_constrained_value(
                model, sigma_ξxsigma_ξx[sigma_ξxsigma_ξx_counter]
            )
            sigma_ξxsigma_ξx_counter += 1
        end
    end
    sigma_ζxsigma_ζx = filter(x -> occursin("sigma_ζ", x), get_names(model))
    sigma_ζxsigma_ζx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, -3)))
        if mod1(i, 3) == 2
            Q_constrained[pos] = get_constrained_value(
                model, sigma_ζxsigma_ζx[sigma_ζxsigma_ζx_counter]
            )
            sigma_ζxsigma_ζx_counter += 1
        end
    end
    sigma_ωxsigma_ωx = filter(x -> occursin("sigma_ω", x), get_names(model))
    sigma_ωxsigma_ωx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, -3)))
        if mod1(i, 3) == 3
            Q_constrained[pos] = get_constrained_value(
                model, sigma_ωxsigma_ωx[sigma_ωxsigma_ωx_counter]
            )
            sigma_ωxsigma_ωx_counter += 1
        end
    end
    Q_unconstrained = cholesky(Symmetric(Q_constrained, :L)).L

    # Updates
    # H
    current_name = 1
    for i in 1:p, j in 1:i
        update_unconstrained_value!(
            model, get_names(model)[current_name], H_unconstrained[i, j]
        )
        current_name += 1
    end
    # Q
    sigma2_ξ_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, 0)))
        if mod1(i, 3) == 1
            update_unconstrained_value!(
                model, sigma2_ξ[sigma2_ξ_counter], Q_unconstrained[pos]
            )
            sigma2_ξ_counter += 1
        end
    end
    sigma2_ζ_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, 0)))
        if mod1(i, 3) == 2
            update_unconstrained_value!(
                model, sigma2_ζ[sigma2_ζ_counter], Q_unconstrained[pos]
            )
            sigma2_ζ_counter += 1
        end
    end
    sigma2_ω_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, 0)))
        if mod1(i, 3) == 3
            update_unconstrained_value!(
                model, sigma2_ω[sigma2_ω_counter], Q_unconstrained[pos]
            )
            sigma2_ω_counter += 1
        end
    end
    sigma_ξxsigma_ξx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, -3)))
        if mod1(i, 3) == 1
            update_unconstrained_value!(
                model, sigma_ξxsigma_ξx[sigma_ξxsigma_ξx_counter], Q_unconstrained[pos]
            )
            sigma_ξxsigma_ξx_counter += 1
        end
    end
    sigma_ζxsigma_ζx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, -3)))
        if mod1(i, 3) == 2
            update_unconstrained_value!(
                model, sigma_ζxsigma_ζx[sigma_ζxsigma_ζx_counter], Q_unconstrained[pos]
            )
            sigma_ζxsigma_ζx_counter += 1
        end
    end
    sigma_ωxsigma_ωx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_unconstrained, -3)))
        if mod1(i, 3) == 3
            update_unconstrained_value!(
                model, sigma_ωxsigma_ωx[sigma_ωxsigma_ωx_counter], Q_unconstrained[pos]
            )
            sigma_ωxsigma_ωx_counter += 1
        end
    end
    return model
end

function fill_model_system!(model::MultivariateBasicStructural)
    Fl = typeof_model_elements(model)
    p = size(model.system.y, 2)
    H_constrained = zeros(Fl, p, p)
    current_name = 1
    for i in 1:p, j in 1:i
        H_constrained[i, j] = get_constrained_value(model, get_names(model)[current_name])
        current_name += 1
    end
    model.system.H = Matrix(Symmetric(H_constrained, :L))
    # Q
    Q_constrained = zeros(Fl, 3p, 3p)
    sigma2_ξ = filter(x -> occursin("sigma2_ξ", x), get_names(model))
    sigma2_ξ_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, 0)))
        if mod1(i, 3) == 1
            Q_constrained[pos] = get_constrained_value(model, sigma2_ξ[sigma2_ξ_counter])
            sigma2_ξ_counter += 1
        end
    end
    sigma2_ζ = filter(x -> occursin("sigma2_ζ", x), get_names(model))
    sigma2_ζ_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, 0)))
        if mod1(i, 3) == 2
            Q_constrained[pos] = get_constrained_value(model, sigma2_ζ[sigma2_ζ_counter])
            sigma2_ζ_counter += 1
        end
    end
    sigma2_ω = filter(x -> occursin("sigma2_ω", x), get_names(model))
    sigma2_ω_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, 0)))
        if mod1(i, 3) == 3
            Q_constrained[pos] = get_constrained_value(model, sigma2_ω[sigma2_ω_counter])
            sigma2_ω_counter += 1
        end
    end
    sigma_ξxsigma_ξx = filter(x -> occursin("sigma_ξ", x), get_names(model))
    sigma_ξxsigma_ξx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, -3)))
        if mod1(i, 3) == 1
            Q_constrained[pos] = get_constrained_value(
                model, sigma_ξxsigma_ξx[sigma_ξxsigma_ξx_counter]
            )
            sigma_ξxsigma_ξx_counter += 1
        end
    end
    sigma_ζxsigma_ζx = filter(x -> occursin("sigma_ζ", x), get_names(model))
    sigma_ζxsigma_ζx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, -3)))
        if mod1(i, 3) == 2
            Q_constrained[pos] = get_constrained_value(
                model, sigma_ζxsigma_ζx[sigma_ζxsigma_ζx_counter]
            )
            sigma_ζxsigma_ζx_counter += 1
        end
    end
    sigma_ωxsigma_ωx = filter(x -> occursin("sigma_ω", x), get_names(model))
    sigma_ωxsigma_ωx_counter = 1
    for (i, pos) in enumerate(collect(diagind(Q_constrained, -3)))
        if mod1(i, 3) == 3
            Q_constrained[pos] = get_constrained_value(
                model, sigma_ωxsigma_ωx[sigma_ωxsigma_ωx_counter]
            )
            sigma_ωxsigma_ωx_counter += 1
        end
    end
    model.system.Q = Matrix(Symmetric(Q_constrained, :L))
    return model
end

function reinstantiate(model::MultivariateBasicStructural, y::Matrix{Fl}) where Fl
    return MultivariateBasicStructural(y, model.seasonality)
end

has_exogenous(::MultivariateBasicStructural) = false
