"""
    kalman_filter(model::StateSpaceModel, H::Matrix{Typ}, Q::Matrix{Typ}; tol::Float64 = 1e-5) where Typ <: AbstractFloat

Kalman filter with big Kappa initialization.
"""
function kalman_filter(model::StateSpaceModel, H::Matrix{Typ}, Q::Matrix{Typ},
                       tol::Typ = 1e-5) where Typ <: AbstractFloat
    if model.mode == "time-variant"
        return time_variant_kalman(model, H, Q)
    else
        return time_invariant_kalman(model, H, Q, tol)
    end
end

function time_invariant_kalman(model::StateSpaceModel, H::Matrix{Typ}, 
                               Q::Matrix{Typ}, tol::Typ) where Typ <: AbstractFloat
    # Load dimensions
    n, p, m, r = size(model)

    # Load system
    y = model.y
    Z, T, R = ztr(model)

    # Predictive state and its covariance
    a = Matrix{Float64}(undef, n+1, m)
    P = Array{Float64, 3}(undef, m, m, n+1)
    att = Matrix{Float64}(undef, n, m)
    Ptt = Array{Float64, 3}(undef, m, m, n)

    # Innovation and its covariance
    v = Matrix{Float64}(undef, n, p)
    F = Array{Float64, 3}(undef, p, p, n)

    # Kalman gain
    K = Array{Float64, 3}(undef, m, p, n)
    
    # Steady state initialization
    steadystate = false
    tsteady     = n+1

    # Initial state: big Kappa initialization
    a[1, :]    = zeros(m, 1)
    P[:, :, 1] = 1e6 .* Matrix(I, m, m)

    RQR = R * Q * R'
    # Kalman filter
    for t = 1:n
        if check_missing_observation(y[t, :])
            steadystate  = false
            v[t, :]      = fill(NaN, p)
            F[:, :, t]   = fill(NaN, (p, p))
            att[t, :]    = a[t, :]
            Ptt[:, :, t] = P[:, :, t]
            a[t+1, :]    = T * att[t, :]
            P[:, :, t+1] = ensure_pos_sym(T * Ptt[:, :, t] * T' + RQR)
        elseif steadystate
            v[t, :]      = y[t, :] - Z[:, :, t] * a[t, :] # v_t = y_t - Z_t * a_t
            F[:, :, t]   = F[:, :, t-1]
            K[:, :, t]   = K[:, :, t-1] # K_t = T * P_t * Z_t * F^-1_t
            att[t, :]    = T*a[t, :] + P[:, :, t] * Z[:, :, t]' * inv(F[:, :, t]) * v[t, :] # att_t = T * a_t + P_t * Z_t' * F^-1_t * v_t
            Ptt[:, :, t] = Ptt[:, :, t-1] # Ptt_t = P_t - P_t * Z_t * F^-1_t * P_t * Z_t
            a[t+1, :]    = T * att[t, :] # a_t+1 = T * att_t
            P[:, :, t+1] = P[:, :, t] # P_t+1 = T * Ptt_t * T' + RQR'
        else
            v[t, :]      = y[t, :] - Z[:, :, t] * a[t, :] # v_t = y_t - Z_t * a_t
            ZP           = Z[:, :, t] * P[:, :, t]
            F[:, :, t]   = ZP * Z[:, :, t]' + H # F_t = Z_t * P_t * Z_t + H
            aux_matrix   = ZP' * inv(F[:, :, t]) 
            K[:, :, t]   = T*aux_matrix # K_t = T * P_t * Z_t * F^-1_t
            att[t, :]    = T*a[t, :] + aux_matrix * v[t, :] # att_t = T * a_t + P_t * Z_t * F^-1_t * v_t
            Ptt[:, :, t] = P[:, :, t] - aux_matrix * ZP # Ptt_t = P_t - P_t * Z_t * F^-1_t * P_t * Z_t
            a[t+1, :]    = T * att[t, :] # a_t+1 = T * att_t
            P[:, :, t+1] = ensure_pos_sym(T * Ptt[:, :, t] * T' + RQR) # P_t+1 = T * Ptt_t * T' + RQR'
            if check_steady_state(P[:, :, t+1], P[:, :, t], tol)
                steadystate = true
                tsteady     = t
            end
        end
    end
    # Return the auxiliary filter structre
    return KalmanFilter(a, att, v, P, Ptt, F, steadystate, tsteady, K)
end

function time_variant_kalman(model::StateSpaceModel, H::Matrix{Typ}, Q::Matrix{Typ}) where Typ <: AbstractFloat
    # Load dimensions
    n, p, m, r = size(model)

    # Load system
    y = model.y
    Z, T, R = ztr(model)

    # Predictive state and its covariance
    a = Matrix{Float64}(undef, n+1, m)
    P = Array{Float64, 3}(undef, m, m, n+1)
    att = Matrix{Float64}(undef, n, m)
    Ptt = Array{Float64, 3}(undef, m, m, n)

    # Innovation and its covariance
    v = Matrix{Float64}(undef, n, p)
    F = Array{Float64, 3}(undef, p, p, n)

    # Kalman gain
    K = Array{Float64, 3}(undef, m, p, n)
    
    # Steady state initialization
    steadystate = false
    tsteady     = n+1

    # Initial state: big Kappa initialization
    a[1, :]    = zeros(m, 1)
    P[:, :, 1] = 1e6 .* Matrix(I, m, m)

    RQR = R * Q * R'
    # Kalman filter
    for t = 1:n
        if check_missing_observation(y[t, :])
            v[t, :]      = fill(NaN, p)
            F[:, :, t]   = fill(NaN, (p, p))
            att[t, :]    = a[t, :]
            Ptt[:, :, t] = P[:, :, t]
            a[t+1, :]    = T * att[t, :]
            P[:, :, t+1] = ensure_pos_sym(T * Ptt[:, :, t] * T' + RQR)
        else
            v[t, :]      = y[t, :] - Z[:, :, t] * a[t, :] # v_t = y_t - Z_t * a_t
            ZP           = Z[:, :, t] * P[:, :, t]
            F[:, :, t]   = ZP * Z[:, :, t]' + H # F_t = Z_t * P_t * Z_t + H
            aux_matrix   = ZP' * inv(F[:, :, t]) 
            K[:, :, t]   = T*aux_matrix # K_t = T * P_t * Z_t * F^-1_t
            att[t, :]    = T*a[t, :] + aux_matrix * v[t, :] # att_t = T * a_t + P_t * Z_t * F^-1_t * v_t
            Ptt[:, :, t] = P[:, :, t] - aux_matrix * ZP # Ptt_t = P_t - P_t * Z_t * F^-1_t * P_t * Z_t
            a[t+1, :]    = T * att[t, :] # a_t+1 = T * att_t
            P[:, :, t+1] = ensure_pos_sym(T * Ptt[:, :, t] * T' + RQR) # P_t+1 = T * Ptt_t * T' + RQR'
        end
    end
    # Return the auxiliary filter structre
    return KalmanFilter(a, att, v, P, Ptt, F, steadystate, tsteady, K)
end

"""
    smoother(model::StateSpaceModel, kfilter::KalmanFilter)

Smoother for state-space model.
"""
function smoother(model::StateSpaceModel, kfilter::KalmanFilter)

    # Load dimensions data
    n, p, m, r = size(model)

    # Load system data
    Z, T, R = ztr(model)

    # Load filter data
    a       = kfilter.a
    v       = kfilter.v
    F       = kfilter.F
    P       = kfilter.P
    K       = kfilter.K

    # Smoothed state and its covariance
    alpha = Matrix{Float64}(undef, n, m)
    V     = Array{Float64, 3}(undef, m, m, n)
    L     = Array{Float64, 3}(undef, m, m, n)
    r     = Matrix{Float64}(undef, n, m)
    N     = Array{Float64, 3}(undef, m, m, n)

    # Initialization
    N[:, :, end] = zeros(m, m)
    r[end, :]    = zeros(m, 1)

    for t = n:-1:2
        if check_missing_observation(v[t, :])
            r[t-1, :]     = T' * r[t, :]
            N[:, :, t-1]  = T' * N[:, :, t] * T
        else
            Z_transp_invF = Z[:, :, t]' * inv(F[:, :, t])
            L[:, :, t]    = T - K[:, :, t] * Z[:, :, t]
            r[t-1, :]     = Z_transp_invF * v[t, :] + L[:, :, t]' * r[t, :]
            N[:, :, t-1]  = Z_transp_invF * Z[:, :, t] + L[:, :, t]' * N[:, :, t] * L[:, :, t]
        end
        alpha[t, :]   = a[t, :] + P[:, :, t] * r[t-1, :]
        V[:, :, t]    = P[:, :, t] - P[:, :, t] * N[:, :, t-1] * P[:, :, t]
    end

    # Last iteration
    if check_missing_observation(v[1, :])
        r0  = T' * r[1, :]
        N0  = T' * N[:, :, 1] * T
    else
        Z_transp_invF = Z[:, :, 1]' * inv(F[:, :, 1])
        L[:, :, 1]    = T - K[:, :, 1] * Z[:, :, 1]
        r0            = Z_transp_invF * v[1, :] + L[:, :, 1]' * r[1, :]
        N0            = Z_transp_invF * Z[:, :, 1] + L[:, :, 1]' * N[:, :, 1] * L[:, :, 1]
    end
    alpha[1, :]   = a[1, :] + P[:, :, 1] * r0
    V[:, :, 1]    = P[:, :, 1] - P[:, :, 1] * N0 * P[:, :, 1]
    # Return the smoothed state structure
    return Smoother(alpha, V)
end

# All filters have to have implemented the following functions
# *
# *
# *

function statespace_covariance(psi::Vector{T}, p::Int, r::Int,
                               filter_type::Type{KalmanFilter}) where T <: AbstractFloat

    # Build lower triangular matrices
    if p > 1
        sqrtH     = tril!(ones(p, p))
        unknownsH = Int(p*(p + 1)/2)
        sqrtH[findall(isequal(1), sqrtH)] = psi[1:unknownsH]
    else
        sqrtH = psi[1].*ones(1, 1)
        unknownsH = 1
    end

    sqrtQ = kron(Matrix{Float64}(I, Int(r/p), Int(r/p)), tril!(ones(p, p)))
    sqrtQ[findall(x -> x == 1, sqrtQ)] = psi[(unknownsH+1):Int(unknownsH + (r/p)*(p*(p + 1)/2))]

    # Obtain full matrices
    H = gram(sqrtH)
    Q = gram(sqrtQ)

    return H, Q
end

function get_log_likelihood_params(psitilde::Vector{T}, model::StateSpaceModel, 
                                   filter_type::Type{KalmanFilter}) where T <: AbstractFloat

    H, Q = statespace_covariance(psitilde, model.dim.p, model.dim.r, filter_type)

    # Obtain innovation v and its variance F
    kfilter = kalman_filter(model, H, Q)
    
    # Return v and F
    return kfilter.v, kfilter.F
end

function kalman_filter_and_smoother(model::StateSpaceModel, covariance::StateSpaceCovariance, 
                                    filter_type::Type{KalmanFilter})

    # Run filter and smoother 
    filtered_state = kalman_filter(model, covariance.H, covariance.Q)
    smoothed_state = smoother(model, filtered_state)
    return FilteredState(filtered_state.att, filtered_state.v, 
                         filtered_state.Ptt, filtered_state.F,
                         filtered_state.steadystate, filtered_state.tsteady),
           SmoothedState(smoothed_state.alpha, smoothed_state.V) 
end