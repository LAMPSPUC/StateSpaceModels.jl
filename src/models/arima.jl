export ARIMA

struct ARIMAOrder
    p::Int
    d::Int
    q::Int
end

# This is used to some optimizations such as
# defining the position of each parameter before hand
# and avoiding string interpolations such as "ar_L$i"
# for "ar_L1"
struct ARIMAHyperParametersAuxiliary
    ar_pos::Vector{Int}
    ar_names::Vector{String}
    ma_pos::Vector{Int}
    ma_names::Vector{String}
    function ARIMAHyperParametersAuxiliary(order::ARIMAOrder)
        return new(ar_position_T_matrix(order), create_ar_names(order),
                   ma_position_R_matrix(order), create_ma_names(order))
    end
end

"""
    ARIMA

An ARIMA model (autoregressive integrated moving average) implemented within the state-space
framework.
"""
mutable struct ARIMA <: StateSpaceModel
    order::ARIMAOrder
    hyperparameters_auxiliary::ARIMAHyperParametersAuxiliary
    hyperparameters::HyperParameters
    system::LinearUnivariateTimeInvariant
    results::Results

    function ARIMA(y::Vector{Fl},
                   order::Tuple{Int, Int, Int}) where Fl

        or = ARIMAOrder(order[1], order[2], order[3])
        hyperparameters_auxiliary = ARIMAHyperParametersAuxiliary(or)

        Z = build_Z(or, Fl)
        T = build_T(or, Fl)
        R = build_R(or, Fl)
        d_ss = zero(Fl)
        c = build_c(or, Fl)
        H = zero(Fl)
        Q_ss = zeros(Fl, 1, 1)

        system = LinearUnivariateTimeInvariant{Fl}(y, Z, T, R, d_ss, c, H, Q_ss)

        num_hyperparameters = or.p + or.q + 1
        names = Vector{String}(undef, num_hyperparameters)
        for i in 1:or.p
            names[i] = "ar_L$i"
        end
        for j in 1:or.q
            names[or.p + j] = "ma_L$j"
        end
        names[end] = "sigma2_η"

        hyperparameters = HyperParameters{Fl}(names)

        return new(or, hyperparameters_auxiliary, hyperparameters, system, Results{Fl}())
    end
end

# Auxiliary functions
function build_Z(or, Fl)
    # Build ARMA Z matrix
    r = max(or.p, or.q + 1)
    arma_Z_matrix = vcat(1, zeros(Fl, r - 1))
    # Build ARIMA Z matrix from ARMA Z matrix
    return vcat(ones(Fl, or.d), arma_Z_matrix)
end
function build_T(or, Fl)
    # Build ARMA T matrix
    r = max(or.p, or.q + 1)
    # zeros matrix in the correct dimension
    arma_T_matrix = zeros(Fl, r, r)
    # Fill ARMA p indices
    for i in 1:or.p
        arma_T_matrix[i, 1] = 1
    end
    for k in 1:r-1
        arma_T_matrix[k, k+1] = 1
    end
    # Build ARIMA T matrix from ARMA T matrix
    arima_T_matrix = zeros(Fl, r + or.d, r + or.d)
    arima_T_matrix[end-r+1:end, end-r+1:end] = arma_T_matrix
    for i in 1:or.d, j in i:or.d+1
        arima_T_matrix[i, j] = 1
    end
    return arima_T_matrix
end
function build_R(or, Fl)
    # Build ARMA R matrix
    r = max(or.p, or.q + 1)
    arma_R_matrix = vcat(ones(1, 1), zeros(Fl, r - 1))
    # Build ARIMA R matrix from ARMA R matrix
    return vcat(zeros(Fl, or.d), arma_R_matrix)
end
function build_c(or, Fl)
    return zeros(Fl, or.d + max(or.p, or.q + 1))
end

function ar_position_T_matrix(or)
    ar_index = CartesianIndex[]
    for i in 1:or.p
        push!(ar_index, CartesianIndex(or.d + i, or.d + 1))
    end
    # To make ARIMA faster we need to rewrite this column wise to access the matrix
    T_dim = max(or.p, or.q + 1) + or.d
    T = zeros(T_dim, T_dim)
    I = LinearIndices(T)
    ar_index_columnwise = Int[]
    for cart_idx in ar_index
        push!(ar_index_columnwise, I[cart_idx])
    end
    return ar_index_columnwise
end
function ma_position_R_matrix(or)
    ma_index = Int[]
    for i in 1:or.q
        push!(ma_index, or.d + i+1)
    end
    return ma_index
end
function create_ar_names(or)
    ar_names = Vector{String}(undef, or.p)
    for i in 1:or.p
        ar_names[i] = "ar_L$i"
    end
    return ar_names
end
function create_ma_names(or)
    ma_names = Vector{String}(undef, or.q)
    for i in 1:or.q
        ma_names[i] = "ma_L$i"
    end
    return ma_names
end
get_ar_name(model::ARIMA, i::Int) = model.hyperparameters_auxiliary.ar_names[i]
get_ma_name(model::ARIMA, i::Int) = model.hyperparameters_auxiliary.ma_names[i]
get_ar_pos(model::ARIMA, i::Int) = model.hyperparameters_auxiliary.ar_pos[i]
get_ma_pos(model::ARIMA, i::Int) = model.hyperparameters_auxiliary.ma_pos[i]

# TODO improve performance possibly preallocating y
#     Monahan, John F. 1984.
#    "A Note on Enforcing Stationarity in
#    Autoregressive-moving Average Models."
#    Biometrika 71 (2) (August 1): 403-404.
function impose_unit_root_constraint(hyperparameter_values::Vector{Fl}) where Fl
    n = length(hyperparameter_values)
    y = fill(zero(Fl), n, n)
    r = @. hyperparameter_values / sqrt((1 + hyperparameter_values^2))
    if n == 1
        return r
    end
    @inbounds for k in 1:n
        for i in 1:k-1
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
    for k in n:-1:2, i in 1:k-1
        y[k - 1, i] = (y[k, i] - y[k, k] * y[k, k - i]) / (1 - y[k, k]^2)
    end
    r = diag(y)
    x = r ./ sqrt.(1 .- r.^2)
    return x
end

function ARIMA_exact_initalization!(filter::KalmanFilter,
                                    model::ARIMA)
    num_arma_states = max(model.order.p, model.order.q + 1)
    ARIMA_exact_initialization!(filter.kalman_state, model.system, num_arma_states)
    return
end
function ARIMA_exact_initialization!(kalman_state, system, num_arma_states)
    arma_T = system.T[end-num_arma_states+1:end, end-num_arma_states+1:end]
    arma_R = system.R[end-num_arma_states+1:end]
    # Calculate exact initial P1
    vec_P1 = (I - kron(arma_T, arma_T)) \ vec(arma_R*arma_R')
    arma_P1 = reshape(vec_P1, num_arma_states, num_arma_states)
    # Fill P1 in the arma states
    kalman_state.P[end-num_arma_states+1:end, end-num_arma_states+1:end] .= system.Q[1] .* arma_P1
    return
end

function update_ar_terms!(model::ARIMA)
    for i in 1:model.order.p
        model.system.T[get_ar_pos(model, i)] = get_constrained_value(model, get_ar_name(model, i))
    end
    return
end
function update_ma_terms!(model::ARIMA)
    for j in 1:model.order.q
        model.system.R[get_ma_pos(model, j)] = get_constrained_value(model, get_ma_name(model, j))
    end
    return
end

# Obligatory functions
function default_filter(model::ARIMA)
    Fl = typeof_model_elements(model)
    num_arma_states = max(model.order.p, model.order.q + 1)
    r = model.order.d + num_arma_states
    a1 = zeros(Fl, r)
    P1 = 1e6 .* Matrix{Fl}(I, r, r)
    skip_llk_instants = model.order.d
    steadystate_tol = Fl(1e-5)
    return UnivariateKalmanFilter(a1, P1, skip_llk_instants, steadystate_tol)
end
function initial_hyperparameters!(model::ARIMA)
    Fl = typeof_model_elements(model)
    # TODO find out how to better estimate initial parameters
    y = filter(!isnan, model.system.y)
    initial_hyperparameters = Dict{String, Fl}(
        "sigma2_η" => dot(y, y)/length(y)
    )
    for i in 1:model.order.p
        initial_hyperparameters[get_ar_name(model, i)] = zero(Fl)
    end
    for j in 1:model.order.q
        initial_hyperparameters[get_ma_name(model, j)] = zero(Fl)
    end
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return
end
function constrain_hyperparameters!(model::ARIMA)
    Fl = typeof_model_elements(model)
    # Constraint AR terms
    unconstrained_ar = Vector{Fl}(undef, model.order.p)
    for i in 1:model.order.p
        unconstrained_ar[i] = get_unconstrained_value(model, get_ar_name(model, i))
    end
    constrained_ar = impose_unit_root_constraint(unconstrained_ar)
    for i in 1:model.order.p
        update_constrained_value!(model, get_ar_name(model, i), constrained_ar[i])
    end

    # Constraint MA terms
    unconstrained_ma = Vector{Fl}(undef, model.order.q)
    for j in 1:model.order.q
        unconstrained_ma[j] = get_unconstrained_value(model, get_ma_name(model, j))
    end
    constrained_ma = -impose_unit_root_constraint(unconstrained_ma)
    for j in 1:model.order.q
        update_constrained_value!(model, get_ma_name(model, j), constrained_ma[j])
    end

    # Constrain variance
    constrain_variance!(model, "sigma2_η")
    return
end
function unconstrain_hyperparameters!(model::ARIMA)
    Fl = typeof_model_elements(model)
    # Unconstraint AR terms
    constrained_ar = Vector{Fl}(undef, model.order.p)
    for i in 1:model.order.p
        constrained_ar[i] = get_constrained_value(model, get_ar_name(model, i))
    end
    unconstrained_ar = relax_unit_root_constraint(constrained_ar)
    for i in 1:model.order.p
        update_unconstrained_value!(model, get_ar_name(model, i), unconstrained_ar[i])
    end

    # Unconstraint MA terms
    constrained_ma = Vector{Fl}(undef, model.order.q)
    for j in 1:model.order.q
        constrained_ma[j] = get_constrained_value(model, get_ma_name(model, j))
    end
    unconstrained_ma = -relax_unit_root_constraint(constrained_ma)
    for j in 1:model.order.q
        update_unconstrained_value!(model, get_ma_name(model, j), unconstrained_ma[j])
    end

    # Unconstrain variance
    unconstrain_variance!(model, "sigma2_η")
    return
end
function fill_model_system!(model::ARIMA)
    update_ar_terms!(model)
    update_ma_terms!(model)
    model.system.Q[1] = get_constrained_value(model, "sigma2_η")
    return nothing
end
function fill_model_filter!(filter::KalmanFilter, model::ARIMA)
    ARIMA_exact_initalization!(filter, model)
    return nothing
end
function reinstantiate(model::ARIMA, y::Vector{Fl}) where Fl
    order = (model.order.p, model.order.d, model.order.q)
    return ARIMA(y, order)
end
