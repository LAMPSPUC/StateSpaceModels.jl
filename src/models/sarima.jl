struct SARIMAOrder
    p::Int
    d::Int
    q::Int
    P::Int
    D::Int
    Q::Int
    s::Int
    r::Int # Model order
    n_states_diff::Int
    n_states::Int
    function SARIMAOrder(p::Int, d::Int, q::Int, P::Int, D::Int, Q::Int, s::Int)
        @assert p >= 0
        @assert d >= 0
        @assert q >= 0
        @assert P >= 0
        @assert D >= 0
        @assert Q >= 0
        @assert s >= 0
        r = max(p + s * P, q + s * Q + 1)
        n_states_diff = d + D * s
        n_states = d + D * s + r
        return new(p, d, q, P, D, Q, s, r, n_states_diff, n_states)
    end
end

# This is used to some optimizations such as
# defining the polynomials beforehand
# and avoiding string interpolations such as "ar_L$i"
# for "ar_L1"
mutable struct SARIMAHyperParametersAuxiliary{Fl <: AbstractFloat}
    ar_poly::Vector{Fl}
    ar_names::Vector{String}
    seasonal_ar_poly::Vector{Fl}
    seasonal_ar_names::Vector{String}
    ma_poly::Vector{Fl}
    ma_names::Vector{String}
    seasonal_ma_poly::Vector{Fl}
    seasonal_ma_names::Vector{String}
    function SARIMAHyperParametersAuxiliary{Fl}(order::SARIMAOrder) where Fl
        ar_poly = create_poly(order.p, Fl)
        seasonal_ar_poly = create_seasonal_poly(order.P, order.s, Fl)
        ma_poly = create_poly(order.q, Fl)
        seasonal_ma_poly = create_seasonal_poly(order.Q, order.s, Fl)
        return new{Fl}(
            ar_poly,
            create_names("ar_L", order.p),
            seasonal_ar_poly,
            create_names("s_ar_L", order.P, order.s),
            ma_poly,
            create_names("ma_L", order.q),
            seasonal_ma_poly,
            create_names("s_ma_L", order.Q, order.s)
        )
    end
end

@doc raw"""
    SARIMA(
        y::Vector{Fl}; 
        order::Tuple{Int,Int,Int} = (1, 0, 0), 
        seasonal_order::Tuple{Int, Int, Int, Int} = (0, 0, 0, 0),
        include_mean::Bool = false
    ) where Fl

A SARIMA model (Seasonal AutoRegressive Integrated Moving Average) implemented within the state-space
framework.

The SARIMA model is specified ``(p, d, q) \times (P, D, Q, s)``. We can also consider a polynomial ``A(t)`` to 
model a trend, here we only allow to add a constante term with the `include_mean` keyword argument.    
```math
\begin{gather*}
    \begin{aligned}
        \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D y_t = A(t) + \theta_q (L) \tilde \theta_Q (L^s) \zeta_t
    \end{aligned}
\end{gather*}
```
In terms of a univariate structural model, this can be represented as
```math
\begin{gather*}
    \begin{aligned}
    y_t & = u_t + \eta_t \\
    \phi_p (L) \tilde \phi_P (L^s) \Delta^d \Delta_s^D u_t & = A(t) + \theta_q (L) \tilde \theta_Q (L^s) \zeta_t
    \end{aligned}
\end{gather*}
```

# Example
```jldoctest
julia> model = SARIMA(rand(100); order=(1,1,1), seasonal_order=(1,2,3,12))
SARIMA(1, 1, 1)x(1, 2, 3, 12) model
```

See more on [Airline passengers](@ref)

# References
 * Durbin, James, & Siem Jan Koopman.
   Time Series Analysis by State Space Methods: Second Edition. 
   Oxford University Press, 2012
"""
mutable struct SARIMA <: StateSpaceModel
    order::SARIMAOrder
    hyperparameters_auxiliary::SARIMAHyperParametersAuxiliary
    hyperparameters::HyperParameters
    system::LinearUnivariateTimeInvariant
    results::Results
    include_mean::Bool

    function SARIMA(y::Vector{Fl}; 
                    order::Tuple{Int,Int,Int} = (1, 0, 0), 
                    seasonal_order::Tuple{Int, Int, Int, Int} = (0, 0, 0, 0),
                    include_mean::Bool = false) where Fl
        or = SARIMAOrder(order[1], order[2], order[3], 
                        seasonal_order[1], seasonal_order[2], seasonal_order[3], seasonal_order[4])
        hyperparameters_auxiliary = SARIMAHyperParametersAuxiliary{Fl}(or)

        Z = build_Z(or, Fl)
        T = build_T(or, Fl)
        R = build_R(or, Fl)
        d_ss = zero(Fl)
        c = build_c(or, Fl)
        H = zero(Fl)
        Q_ss = zeros(Fl, 1, 1)

        system = LinearUnivariateTimeInvariant{Fl}(y, Z, T, R, d_ss, c, H, Q_ss)

        num_hyperparameters = or.p + or.q + or.P + or.Q + 1
        if include_mean 
            num_hyperparameters += 1
        end
        names = Vector{String}(undef, num_hyperparameters)
        for i in 1:(or.p)
            names[i] = "ar_L$i"
        end
        for j in 1:(or.q)
            names[or.p + j] = "ma_L$j"
        end
        for is in 1:(or.P)
            names[or.p + or.q + is] = "s_ar_L$(or.s * is)"
        end
        for js in 1:(or.Q)
            names[or.p + or.q + or.P + js] = "s_ma_L$(or.s * js)"
        end
        if include_mean
            names[end-1] = "mean"
        end
        names[end] = "sigma2_η"

        hyperparameters = HyperParameters{Fl}(names)

        return new(or, hyperparameters_auxiliary, hyperparameters, system, Results{Fl}(), include_mean)
    end
end

# Auxiliary functions
function build_Z(or, Fl)
    seasonal_entries = or.s - 1 >= 0 ? or.s - 1 : 0
    return vcat(
            fill(one(Fl), or.d),
            repeat(vcat(zeros(Fl, seasonal_entries), one(Fl)), or.D),
            one(Fl),
            fill(zero(Fl), or.r - 1)
        )
end

triu_indices(k::Int) = findall(isone, triu(ones(k, k)))
function build_T(or, Fl)
    # zeros matrix in the correct dimension
    T = zeros(Fl, or.n_states, or.n_states)
    if or.r > 0
        T[end - or.r + 1:end, end - or.r + 1:end] = diagm(1 => ones(or.r - 1))
    end
    if or.D > 0
        seasonal_companion = diagm(-1 => ones(or.s - 1))
        seasonal_companion[1, end] = 1
        for l in 1:or.D
            start_idx = or.d + (l - 1) * or.s
            end_idx = or.d + l * or.s
            T[start_idx + 1:end_idx, start_idx + 1:end_idx] .= seasonal_companion
            if l + 1 <= or.D
                T[start_idx + 1, end_idx + or.s] = 1
            end
            T[start_idx + 1, or.n_states_diff + 1] = 1
        end
    end
    if or.d > 0
        idx = triu_indices(or.d)
        T[idx] .= 1
        if or.s > 0
            start_idx = or.d
            end_idx = or.n_states_diff
            T[1:or.d, start_idx+1:end_idx] .= repeat(hcat(zeros(Fl, 1, or.s - 1), one(Fl)), 1, or.D)
        end
        T[1:or.d, or.n_states_diff + 1] .= 1
    end
    return T
end
function build_R(or, Fl)
    return vcat(
        fill(zero(Fl), or.n_states_diff, 1),
        fill(one(Fl), 1, 1),
        fill(zero(Fl), or.r - 1, 1)
    )
end
function build_c(or, Fl)
    return zeros(Fl, or.n_states)
end

function create_names(str::String, dim::Int, s::Int = 1)
    names = Vector{String}(undef, dim)
    for i in 1:dim
        names[i] = str*"$(s * i)"
    end
    return names
end
function create_poly(n::Int, Fl::DataType)
    return fill(one(Fl), n + 1)
end
function create_seasonal_poly(n::Int, s::Int, Fl::DataType)
    seasonal_poly = fill(zero(Fl), (s * n) + 1)
    seasonal_poly[1] = one(Fl)
    for i in 1:n
        seasonal_poly[(i * s) + 1] = one(Fl)
    end
    return seasonal_poly
end
function poly_mul(p1::Vector{Fl}, p2::Vector{Fl}) where Fl
    n, m = length(p1), length(p2)
    if n > 1 && m > 1
        c = zeros(Fl, m + n - 1)
        for i in 1:n, j in 1:m
            c[i + j - 1] += p1[i] * p2[j]
        end
        return c
    elseif n <= 1
        return p2
    else
        return p1
    end
end
get_ar_name(model::SARIMA, i::Int) = model.hyperparameters_auxiliary.ar_names[i]
get_ma_name(model::SARIMA, i::Int) = model.hyperparameters_auxiliary.ma_names[i]
get_seasonal_ar_name(model::SARIMA, i::Int) = model.hyperparameters_auxiliary.seasonal_ar_names[i]
get_seasonal_ma_name(model::SARIMA, i::Int) = model.hyperparameters_auxiliary.seasonal_ma_names[i]

function update_T_terms!(model::SARIMA)
    update_ar_poly!(model)
    update_seasonal_ar_poly!(model)
    # TODO increaase performance
    # We could preallocate this polynomial p3
    p3 = poly_mul(model.hyperparameters_auxiliary.ar_poly, model.hyperparameters_auxiliary.seasonal_ar_poly)
    for i in model.order.n_states_diff + 1:model.order.n_states_diff + model.order.p + model.order.s * model.order.P
        model.system.T[i, model.order.n_states_diff + 1] = p3[i - model.order.n_states_diff + 1]
    end
    return nothing
end

function update_R_terms!(model::SARIMA)
    update_ma_poly!(model)
    update_seasonal_ma_poly!(model)
    # TODO increaase performance
    # We could preallocate this polynomial p3
    p3 = poly_mul(model.hyperparameters_auxiliary.ma_poly, model.hyperparameters_auxiliary.seasonal_ma_poly)
    for i in model.order.n_states_diff + 1:model.order.n_states_diff + model.order.q + model.order.s * model.order.Q
        model.system.R[i + 1] = p3[i - model.order.n_states_diff + 1]
    end
    return nothing
end

function update_ar_poly!(model::SARIMA)
    if model.order.p > 0
        for i in 1:(model.order.p)
            model.hyperparameters_auxiliary.ar_poly[i + 1] = get_constrained_value(
                model, get_ar_name(model, i)
            )
        end
    end
end
function update_seasonal_ar_poly!(model::SARIMA)
    if model.order.P > 0
        for i in 1:(model.order.P)
            model.hyperparameters_auxiliary.seasonal_ar_poly[(i * model.order.s) + 1] = get_constrained_value(
                model, get_seasonal_ar_name(model, i)
            )
        end
    end
end
function update_ma_poly!(model::SARIMA)
    if model.order.q > 0
        for i in 1:(model.order.q)
            model.hyperparameters_auxiliary.ma_poly[i + 1] = get_constrained_value(
                model, get_ma_name(model, i)
            )
        end
    end
end
function update_seasonal_ma_poly!(model::SARIMA)
    if model.order.Q > 0
        for i in 1:(model.order.Q)
            model.hyperparameters_auxiliary.seasonal_ma_poly[(i * model.order.s) + 1] = get_constrained_value(
                model, get_seasonal_ma_name(model, i)
            )
        end
    end
end


function impose_unit_root_constraint(hyperparameter_values::Vector{Fl}) where Fl
    # TODO improve performance possibly preallocating y
    #     Monahan, John F. 1984.
    #    "A Note on Enforcing Stationarity in
    #    Autoregressive-moving Average Models."
    #    Biometrika 71 (2) (August 1): 403-404.
    n = length(hyperparameter_values)
    y = fill(zero(Fl), n, n)
    r = @. hyperparameter_values / sqrt((1 + hyperparameter_values^2))
    if n == 1
        return -r
    end
    @inbounds for k in 1:n
        for i in 1:(k - 1)
            y[k, i] = y[k - 1, i] + r[k] * y[k - 1, k - i]
        end
        y[k, k] = r[k]
    end
    return -@view y[n, :]
end

function relax_unit_root_constraint(hyperparameter_values::Vector{Fl}) where Fl
    #     Monahan, John F. 1984.
    #    "A Note on Enforcing Stationarity in
    #    Autoregressive-moving Average Models."
    #    Biometrika 71 (2) (August 1): 403-404.
    n = length(hyperparameter_values)
    y = fill(zero(Fl), n, n)
    y[n, :] = -hyperparameter_values
    for k in n:-1:2, i in 1:(k - 1)
        y[k - 1, i] = (y[k, i] - y[k, k] * y[k, k - i]) / (1 - y[k, k]^2)
    end
    r = diag(y)
    x = r ./ sqrt.(1 .- r .^ 2)
    return x
end

function SARIMA_exact_initialization!(filter::KalmanFilter, model::SARIMA)
    SARIMA_exact_initialization!(filter.kalman_state, model.system, model.order.r)
    return nothing
end

function SARIMA_exact_initialization!(kalman_state, 
                                      system,
                                      num_arma_states::Int)
    # TODO we could preallocate these arrays
    arma_T = system.T[(end - num_arma_states + 1):end, (end - num_arma_states + 1):end]
    arma_R = system.R[(end - num_arma_states + 1):end]
    # Calculate exact initial P1
    arma_P1 = lyapd(arma_T, arma_R * arma_R')
    # Fill P1 in the arma states
    kalman_state.P[(end - num_arma_states + 1):end, (end - num_arma_states + 1):end] .=
        system.Q[1] .* arma_P1
    return nothing
end

function concatenate_on_bottom(X1::Matrix{Fl}, X2::Matrix{Fl}) where Fl
    n = min(size(X1, 1), size(X2, 1))
    return hcat(X1[end-n+1:end, :], X2[end-n+1:end, :])
end

function diff_sarima(y::Vector{Fl}, d::Int, D::Int, s::Int) where Fl
    # Seasonal differencing
    for _ in 1:D
        y = y[s:end] - y[1:end-s+1]
    end
    # Simple differencing
    for _ in 1:d
        y = diff(y)
    end
    return y
end

function constrain_ar!(model::SARIMA, Fl::DataType)
    if model.order.p > 0
        unconstrained_ar = Vector{Fl}(undef, model.order.p)
        for i in 1:(model.order.p)
            unconstrained_ar[i] = get_unconstrained_value(model, get_ar_name(model, i))
        end
        constrained_ar = impose_unit_root_constraint(unconstrained_ar)
        for i in 1:(model.order.p)
            update_constrained_value!(model, get_ar_name(model, i), constrained_ar[i])
        end
    end
    return nothing
end
function constrain_ma!(model::SARIMA, Fl::DataType)
    if model.order.q > 0
        unconstrained_ma = Vector{Fl}(undef, model.order.q)
        for j in 1:(model.order.q)
            unconstrained_ma[j] = get_unconstrained_value(model, get_ma_name(model, j))
        end
        constrained_ma = -impose_unit_root_constraint(unconstrained_ma)
        for j in 1:(model.order.q)
            update_constrained_value!(model, get_ma_name(model, j), constrained_ma[j])
        end
    end
    return nothing
end
function constrain_seasonal_ar!(model::SARIMA, Fl::DataType)
    if model.order.P > 0
        unconstrained_seasonal_ar = Vector{Fl}(undef, model.order.P)
        for i in 1:(model.order.P)
            unconstrained_seasonal_ar[i] = get_unconstrained_value(model, get_seasonal_ar_name(model, i))
        end
        constrained_seasonal_ar = impose_unit_root_constraint(unconstrained_seasonal_ar)
        for i in 1:(model.order.P)
            update_constrained_value!(model, get_seasonal_ar_name(model, i), constrained_seasonal_ar[i])
        end
    end
    return nothing
end
function constrain_seasonal_ma!(model::SARIMA, Fl::DataType)
    if model.order.Q > 0
        unconstrained_seasonal_ma = Vector{Fl}(undef, model.order.Q)
        for j in 1:(model.order.Q)
            unconstrained_seasonal_ma[j] = get_unconstrained_value(model, get_seasonal_ma_name(model, j))
        end
        constrained_seasonal_ma = -impose_unit_root_constraint(unconstrained_seasonal_ma)
        for j in 1:(model.order.Q)
            update_constrained_value!(model, get_seasonal_ma_name(model, j), constrained_seasonal_ma[j])
        end
    end
    return nothing
end
function constrain_mean!(model::SARIMA)
    if model.include_mean
        constrain_identity!(model, "mean")
    end
    return nothing
end

function unconstrain_ar!(model::SARIMA, Fl::DataType)
    if model.order.p > 0
        constrained_ar = Vector{Fl}(undef, model.order.p)
        for i in 1:(model.order.p)
            constrained_ar[i] = get_constrained_value(model, get_ar_name(model, i))
        end
        unconstrained_ar = relax_unit_root_constraint(constrained_ar)
        for i in 1:(model.order.p)
            update_unconstrained_value!(model, get_ar_name(model, i), unconstrained_ar[i])
        end
    end
    return nothing
end
function unconstrain_ma!(model::SARIMA, Fl::DataType)
    if model.order.q > 0
        constrained_ma = Vector{Fl}(undef, model.order.q)
        for j in 1:(model.order.q)
            constrained_ma[j] = get_constrained_value(model, get_ma_name(model, j))
        end
        unconstrained_ma = -relax_unit_root_constraint(constrained_ma)
        for j in 1:(model.order.q)
            update_unconstrained_value!(model, get_ma_name(model, j), unconstrained_ma[j])
        end
    end
    return nothing
end
function unconstrain_seasonal_ar!(model::SARIMA, Fl::DataType)
    if model.order.P > 0
        constrained_seasonal_ar = Vector{Fl}(undef, model.order.P)
        for i in 1:(model.order.P)
            constrained_seasonal_ar[i] = get_constrained_value(model, get_seasonal_ar_name(model, i))
        end
        unconstrained_seasonal_ar = relax_unit_root_constraint(constrained_seasonal_ar)
        for i in 1:(model.order.P)
            update_unconstrained_value!(model, get_seasonal_ar_name(model, i), unconstrained_seasonal_ar[i])
        end
    end
    return nothing
end
function unconstrain_seasonal_ma!(model::SARIMA, Fl::DataType)
    if model.order.Q > 0
        constrained_seasonal_ma = Vector{Fl}(undef, model.order.Q)
        for j in 1:(model.order.Q)
            constrained_seasonal_ma[j] = get_constrained_value(model, get_seasonal_ma_name(model, j))
        end
        unconstrained_seasonal_ma = -relax_unit_root_constraint(constrained_seasonal_ma)
        for j in 1:(model.order.Q)
            update_unconstrained_value!(model, get_seasonal_ma_name(model, j), unconstrained_seasonal_ma[j])
        end
    end
    return nothing
end
function unconstrain_mean!(model::SARIMA)
    if model.include_mean
        unconstrain_identity!(model, "mean")
    end
    return nothing
end

function ar_polinomial(p::Vector{Fl}) where Fl
    return Polynomial([one(Fl); -p])
end

function ma_polinomial(q::Vector{Fl}) where Fl
    return Polynomial([one(Fl); q])
end

function roots_of_inverse_polinomial(poly::Polynomial)
    return roots(poly).^-1
end

function assert_stationarity(p::Vector{Fl}) where Fl
    poly = ar_polinomial(p)
    return all(abs.(roots_of_inverse_polinomial(poly)) .< 1)
end

function assert_invertibility(q::Vector{Fl}) where Fl
    poly = ma_polinomial(q)
    return all(abs.(roots_of_inverse_polinomial(poly)) .< 1)
end

function conditional_sum_of_squares(y_diff::Vector{Fl}, k_ar::Int, k_ma::Int) where Fl
    if (k_ar == 0) && (k_ma == 0)
        return (Fl[], Fl[])
    end
    k = 2 * k_ma
    r = max(k + k_ma, k_ar)
    residuals = nothing
    X = nothing
    if k_ar + k_ma > 0
        # If we have MA terms, get residuals from an AR(k) model to use
        # as data for conditional sum of squares estimates of the MA
        # parameters
        if k_ma > 0
            y = y_diff[k:end]
            X = lagmat(y_diff, k)
            params_ar = X \ y[end - size(X, 1) + 1:end]
            residuals = y[end - size(X, 1) + 1:end] - X * params_ar
        end
        # Run an ARMA(p,q) model using the just computed residuals as
        # data
        y = y_diff[r:end]
        X = Matrix{Fl}(undef, length(y), 0)
        X = concatenate_on_bottom(X, lagmat(y_diff, k_ar))
        if k_ma > 0
            X = concatenate_on_bottom(X, lagmat(residuals, k_ma))
        end
    end

    initial_params = X \ y_diff[end - size(X, 1) + 1:end]
    params_ar = Fl[]
    params_ma = Fl[]
    offset = 1
    if k_ar > 0
        params_ar = initial_params[offset:offset + k_ar - 1]
        offset += k_ar
    end
    if k_ma > 0
        params_ma = initial_params[offset:offset + k_ma - 1]
    end
    return params_ar, params_ma
end

# Obligatory functions
function default_filter(model::SARIMA)
    Fl = typeof_model_elements(model)
    a1 = zeros(Fl, model.order.n_states)
    P1 = Fl(1e6) .* Matrix{Fl}(I, model.order.n_states, model.order.n_states)
    skip_llk_instants = model.order.d + model.order.s * model.order.D
    steadystate_tol = Fl(1e-5)
    return UnivariateKalmanFilter(a1, P1, skip_llk_instants, steadystate_tol)
end

function initial_hyperparameters!(model::SARIMA)
    Fl = typeof_model_elements(model)
    initial_hyperparameters = Dict{String,Fl}()
    # Heuristic inspired in statsmodels from python
    # TODO find a reference to this heuristic
    # conditional sum of squares
    y_diff = diff_sarima(model.system.y, model.order.d, model.order.D, model.order.s)
    y_diff = filter(!isnan, y_diff)
    # Non-seasonal ARMA
    (initial_ar, initial_ma) = conditional_sum_of_squares(y_diff, model.order.p, model.order.q)
    if !assert_stationarity(initial_ar)
        @warn("Conditional sum of squares estimated initial_ar out of the unit circle, using zero as starting params")
        initial_ar .= zero(Fl)
    end
    if !assert_invertibility(initial_ma)
        @warn("Conditional sum of squares estimated initial_ma out of the unit circle, using zero as starting params")
        initial_ma .= zero(Fl)
    end
    # Seasonal ARMA
    (initial_seasonal_ar, 
    initial_seasonal_ma) = conditional_sum_of_squares(y_diff, model.order.P, model.order.Q)
    if !assert_stationarity(initial_seasonal_ar)
        @warn("Conditional sum of squares estimated initial_seasonal_ar out of the unit circle, using zero as starting params")
        initial_seasonal_ar .= zero(Fl)
    end
    if !assert_invertibility(initial_seasonal_ma)
        @warn("Conditional sum of squares estimated initial_seasonal_ma out of the unit circle, using zero as starting params")
        initial_seasonal_ma .= zero(Fl)
    end
    for i in 1:(model.order.p)
        initial_hyperparameters[get_ar_name(model, i)] = initial_ar[i]
    end
    for j in 1:(model.order.q)
        initial_hyperparameters[get_ma_name(model, j)] = initial_ma[j]
    end
    for i in 1:(model.order.P)
        initial_hyperparameters[get_seasonal_ar_name(model, i)] = initial_seasonal_ar[i]
    end
    for j in 1:(model.order.Q)
        initial_hyperparameters[get_seasonal_ma_name(model, j)] = initial_seasonal_ma[j]
    end
    if model.include_mean 
        initial_hyperparameters["mean"] = mean(y_diff)
    end
    
    initial_hyperparameters["sigma2_η"] = var(y_diff)
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return nothing
end

function constrain_hyperparameters!(model::SARIMA)
    Fl = typeof_model_elements(model)
    constrain_ar!(model, Fl)
    constrain_ma!(model, Fl)
    constrain_seasonal_ar!(model, Fl)
    constrain_seasonal_ma!(model, Fl)
    constrain_mean!(model)
    constrain_variance!(model, "sigma2_η")
    return nothing
end

function unconstrain_hyperparameters!(model::SARIMA)
    Fl = typeof_model_elements(model)
    unconstrain_ar!(model, Fl)
    unconstrain_ma!(model, Fl)
    unconstrain_seasonal_ar!(model, Fl)
    unconstrain_seasonal_ma!(model, Fl)
    unconstrain_mean!(model)
    unconstrain_variance!(model, "sigma2_η")
    return nothing
end

function fill_model_system!(model::SARIMA)
    update_T_terms!(model)
    update_R_terms!(model)
    if model.include_mean
        model.system.d = get_constrained_value(model, "mean")
    end
    model.system.Q[1] = get_constrained_value(model, "sigma2_η")
    return nothing
end

function fill_model_filter!(filter::KalmanFilter, model::SARIMA)
    SARIMA_exact_initialization!(filter, model)
    return nothing
end

function reinstantiate(model::SARIMA, y::Vector{Fl}) where Fl
    return SARIMA(y; 
                order = (model.order.p, model.order.d, model.order.q), 
                seasonal_order = (model.order.P, model.order.D, model.order.Q, model.order.s), 
                include_mean = model.include_mean
            )
end

has_exogenous(::SARIMA) = false

function model_name(model::SARIMA)
    p, d, q = model.order.p, model.order.d, model.order.q
    P, D, Q, s = model.order.P, model.order.D, model.order.Q, model.order.s
    str = "SARIMA($p, $d, $q)x($P, $D, $Q, $s)"
    if model.include_mean 
        str *= " with non-zero mean"
    end
    return str
end

# autoarima functions
abstract type UnitRootTests end

struct KPSS <: UnitRootTests end
struct ADF <: UnitRootTests end


function assert_ic(ic::Symbol)
    @assert ic in [:AIC, :AICc, :BIC]
    return 
end

function assert_test(test::Symbol)
    @assert test in [:KPSS, :ADF]
    return
end

function assert_non_negativity(order::Int)
    @assert order >= 0
    return
end

function assert_frequency(frequency, seasonal::Bool)
    if seasonal 
        @assert frequency > 0
    end
    return
end

function assert_parameters(frequency,
                            d,
                            D,
                            max_p::Int,
                            max_q::Int,
                            max_P::Int,
                            max_Q::Int,
                            max_d::Int,
                            max_D::Int,
                            start_p::Int,
                            start_q::Int,
                            start_P::Int,
                            start_Q::Int,
                            seasonal::Bool,
                            ic::Symbol,
                            test::Symbol,
                            )
    assert_frequency(frequency, seasonal)
    isnan(d) ? nothing : assert_non_negativity(d)
    isnan(D) ? nothing : assert_non_negativity(D)
    assert_non_negativity(max_p)
    assert_non_negativity(max_q)
    assert_non_negativity(max_P)
    assert_non_negativity(max_Q)
    assert_non_negativity(max_d)
    assert_non_negativity(max_D)
    assert_non_negativity(start_p)
    assert_non_negativity(start_q)
    assert_non_negativity(start_P)
    assert_non_negativity(start_Q)
    assert_test(test)
    assert_ic(ic)

    return 
end

function handle_seasonal_integration_order(y::Vector{Float64}, max_D::Int, D)
    
    return !isnan(D) ? D : handle_seasonal_integration_order(y, max_D)
end

function handle_seasonal_integration_order(y::Vector{Float64}, max_D::Int)
    #todo 
    
    return 0
end

function handle_seasonal_diff(y::Vector{Float64}, frequency::Int)
    
    return y[frequency + 1:end] .- y[1:end - frequency]
end

function handle_test_args(test_args::Dict)
    if isempty(test_args)
        test_args[:null] = :level
        test_args[:lags] = true
    end

    return test_args
end

# function handle_test_args(test_args::Dict, y::Vector{Float64})
#     if isempty(test_args)
#         test_args[:null] = :constant
#         test_args[:lags] = trunc(Int64, (length(y) - 1)^(1/3))
#     end
#     return test_args
# end

function handle_integration_order(y::Vector{Float64}, max_d::Int, D::Int, frequency, test_args::Dict, ::Type{KPSS})
    test_parms = handle_test_args(test_args)
    seasonal_diff_y = D > 0 ? handle_seasonal_diff(y, frequency) : y
    p_value = KPSSTest(seasonal_diff_y; null = test_parms[:null], lags = test_parms[:lags])
    d = 0
    
    while p_value <= 0.01 && d < max_d
       seasonal_diff_y = diff(seasonal_diff_y)
       p_value = KPSSTest(seasonal_diff_y; null = test_parms[:null], lags = test_parms[:lags])   
       d += 1
    end
    
    return d
end

function handle_integration_order(y::Vector{Float64}, max_d::Int, D::Int, frequency, test_args::Dict, test::Symbol)
    unit_root_type = handle_unit_root_type(test)
    return handle_integration_order(y, max_d, D, frequency, test_args, unit_root_type)
end

# function handle_integration_order(y::Vector{Float64}, max_d::Int, D::Int, frequency::Int, test_args::Dict, ::Type{ADF})
#     test_parms = handle_test_args(test_args, y)
#     seasonal_diff_y = D > 0 ? handle_seasonal_diff(y, frequency) : y
#     p_value = ADFTest(y, test_parms[:null], test_parms[:lags])
#     d = 0
    
#     while p_value > 0.1 && d < max_d
#        seasonal_diff_y = diff(seasonal_diff_y)
#        p_value = ADFTest(seasonal_diff_y, test_parms[:null], test_parms[:lags])   
#        d += 1
#     end
    
#     return d
# end

function handle_unit_root_type(test::Symbol)
    return test == :KPSS ? KPSS : ADF
end

mutable struct AutoARIMAParameters
    frequency::Int
    d::Int
    D::Int
    max_p::Int
    max_q::Int
    max_P::Int
    max_Q::Int
    max_d::Int
    max_D::Int
    stationary::Bool
    ic::Symbol               
    nmodels::Int
    trace::Bool
    allow_intercept::Bool
end

"""
    Explicar certin
"""
function autoarima(y::Vector{Fl};
                   frequency = NaN,
                   d = NaN,
                   D = NaN,
                   max_p::Int = 5,
                   max_q::Int = 5,
                   max_P::Int = 2,
                   max_Q::Int = 2,
                   max_d::Int = 2,
                   max_D::Int = 1,
                   start_p::Int = 2,
                   start_q::Int = 2,
                   start_P::Int = 1,
                   start_Q::Int = 1,
                   stationary::Bool = false,
                   seasonal::Bool = false,
                   ic::Symbol = :AIC,
                   stepwise::Bool = true,
                   nmodels::Int = 94, #From R
                   trace::Bool = true,
                   # approximation,
                   # method,
                   truncate = nothing, #must be Int
                   # xreg,
                   test::Symbol = :KPSS,
                   test_args::Dict = Dict(),
                   # seasonal_test,
                   # seasonal_test_args,
                   allowdrift = true,
                   allowmean = true
                   # lambda = NULL,
                   # biasadj = FALSE,
                   # parallel = FALSE,
                   # num.cores = 2,         
                   ) where Fl
    
    # Asserts para verificar se ic e test são algum dos implementados
    assert_parameters(
        frequency, d, D,
        max_p, max_q, max_P, max_Q, max_d, max_D, 
        start_p, start_q, start_P, start_Q, seasonal,
        ic, test
    )
    # Definir D
    D = seasonal ? handle_seasonal_integration_order(y, max_D, D) : 0
    # Definir d
    d = handle_integration_order(y, max_d, D, frequency, test_args, test)
    # if stepwise == true
        # Definir o primeiro current model

        # Criação do conjunto de modelos baseado no current model
        
        # Estimar os modelos do conjunto

        # Atualizar o current model
        
    # elseif stepwise == false
        # Escolher ordem na força bruta dentre todos os
        # parâmetros permitidos
    # end
    return D, d
end