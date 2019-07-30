"""
    kalman_filter(model::StateSpaceModel, H::Matrix{Typ}, Q::Matrix{Typ}; tol::Float64 = 1e-5) where Typ <: AbstractFloat

Kalman filter with big Kappa initialization.
"""
function kalman_filter(model::StateSpaceModel, H::Matrix{Typ}, Q::Matrix{Typ}; tol::Typ = 1e-5) where Typ <: AbstractFloat

    time_invariant = model.mode == "time-invariant"

    # Load system
    Z, T, R = ztr(model)

    # Predictive state and its covariance
    a = Matrix{Float64}(undef, model.dim.n+1, model.dim.m)
    P = Array{Float64, 3}(undef, model.dim.m, model.dim.m, model.dim.n+1)
    att = Matrix{Float64}(undef, model.dim.n, model.dim.m)
    Ptt = Array{Float64, 3}(undef, model.dim.m, model.dim.m, model.dim.n)

    # Innovation and its covariance
    v = Matrix{Float64}(undef, model.dim.n, model.dim.p)
    F = Array{Float64, 3}(undef, model.dim.p, model.dim.p, model.dim.n)

    # Kalman gain
    K = Array{Float64, 3}(undef, model.dim.m, model.dim.p, model.dim.n)
    
    # Auxiliary structures
    ZP = Array{Float64, 2}(undef, model.dim.p, model.dim.m)
    P_Ztransp_invF = Array{Float64, 2}(undef, model.dim.m, model.dim.p)
    
    # Steady state initialization
    steadystate = false
    tsteady     = model.dim.n+1

    # Initial state: big Kappa initialization
    a[1, :]    = zeros(model.dim.m, 1)
    P[:, :, 1] = 1e6 .* Matrix(I, model.dim.m, model.dim.m)

    RQR = R * Q * R'
    # Kalman filter
    for t = 1:model.dim.n
        if check_missing_observation(model.y, t)
            steadystate  = false
            v[t, :]      = fill(NaN, model.dim.p)
            F[:, :, t]   = fill(NaN, (model.dim.p, model.dim.p))
            att[t, :]    = a[t, :]
            Ptt[:, :, t] = P[:, :, t]
            update_a(a, att, T, t) # a_t+1 = T * att_t
            update_P(P, T, Ptt, RQR, t)
        elseif steadystate
            update_v(v, model.y, Z, a, t) # v_t = y_t - Z_t * a_t
            repeat_matrix_t_plus_1(F, t-1) # F[:, :, t]   = F[:, :, t-1]
            repeat_matrix_t_plus_1(K, t-1) # K[:, :, t]   = K[:, :, t-1]
            att[t, :]    = a[t, :] + P[:, :, t] * Z[:, :, t]' * invF(F, t) * v[t, :] # att_t = a_t + P_t * Z_t' * F^-1_t * v_t
            repeat_matrix_t_plus_1(Ptt, t-1) # Ptt_t = Ptt_t-1
            update_a(a, att, T, t) # a_t+1 = T * att_t
            repeat_matrix_t_plus_1(P, t) # P_t+1 = P_t
        else
            update_v(v, model.y, Z, a, t) # v_t = y_t - Z_t * a_t
            update_ZP(ZP, Z, P, t) # ZP = Z[:, :, t] * P[:, :, t]
            update_F(F, ZP, Z, H, t) # F_t = Z_t * P_t * Z_t + H
            update_P_Ztransp_Finv(P_Ztransp_invF, ZP, F, t) # P_Ztransp_invF   = ZP' * invF(F, t)
            update_K(K, P_Ztransp_invF, T, t) # K_t = T * P_t * Z_t * F^-1_t
            update_att(att, a, P_Ztransp_invF, v, t) # att_t = a_t + P_t * Z_t * F^-1_t * v_t
            Ptt[:, :, t] = P[:, :, t] - P_Ztransp_invF * ZP # Ptt_t = P_t - P_t * Z_t * F^-1_t * P_t * Z_t
            update_a(a, att, T, t) # a_t+1 = T * att_t
            update_P(P, T, Ptt, RQR, t) # P_t+1 = T * Ptt_t * T' + RQR'
            if check_steady_state(P, t, tol) && time_invariant
                steadystate = true
                tsteady     = t
            end
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
        if check_missing_observation(v, t)
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
    if check_missing_observation(v, 1)
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

# All filters have to implement the following functions
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
    sqrtQ[findall(isequal(1), sqrtQ)] = psi[(unknownsH+1):Int(unknownsH + (r/p)*(p*(p + 1)/2))]

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
    return FilterOutput(filtered_state.a, filtered_state.att, filtered_state.v, 
                        filtered_state.P, filtered_state.Ptt, filtered_state.F,
                        filtered_state.steadystate, filtered_state.tsteady),
           SmoothedState(smoothed_state.alpha, smoothed_state.V) 
end

# Utils for filter performance
function repeat_matrix_t_plus_1(mat::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @inbounds for j in axes(mat, 2), i in axes(mat, 1)
          mat[i, j, t+1] = mat[i, j, t]
    end
    return 
end

function update_a(a::Matrix{Typ}, att::Matrix{Typ}, T::Matrix{Typ}, t::Int) where Typ <: AbstractFloat
    @views @inbounds mul!(a[t+1, :], T, att[t, :])
    return 
end

function update_ZP(ZP::AbstractArray{T}, Z::AbstractArray{T}, P::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @inbounds @views mul!(ZP, Z[:, :, t], P[:, :, t])
    return 
end

function update_K(K::AbstractArray{Typ}, P_Ztransp_invF::AbstractArray{Typ}, T::AbstractArray{Typ}, t::Int) where Typ <: AbstractFloat
    @inbounds @views mul!(K[:, :, t], T, P_Ztransp_invF)
    return
end

function invF(F::AbstractArray{T}, t::Int) where T
    return @inbounds @views inv(F[:, :, t]) 
end

function update_P(P::AbstractArray{Typ}, T::AbstractArray{Typ}, Ptt::AbstractArray{Typ}, RQR::AbstractArray{Typ}, t::Int)  where Typ <: AbstractFloat 
    @views @inbounds LinearAlgebra.BLAS.gemm!('N', 'T', 1.0, T * Ptt[:, :, t], T, 0.0, P[:, :, t+1]) # P[:, :, t+1] = T * Ptt[:, :, t] * T'
    sum_matrix(P, RQR, t, 1) # P[:, :, t+1] .+= RQR
    ensure_pos_sym(P, t + 1)
    return 
end

function update_v(v::AbstractArray{T}, y::AbstractArray{T}, Z::AbstractArray{T}, a::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @views @inbounds  v[t, :] = y[t, :] - Z[:, :, t] * a[t, :]
    return 
end

function update_att(att::AbstractArray{T}, a::AbstractArray{T}, P_Ztransp_invF::AbstractArray{T}, v::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @inbounds @views att[t, :] = a[t, :] + P_Ztransp_invF * v[t, :]
    return 
end

function update_F(F::AbstractArray{T}, ZP::AbstractArray{T}, Z::AbstractArray{T}, H::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @views @inbounds LinearAlgebra.BLAS.gemm!('N', 'T', 1.0, ZP, Z[:, :, t], 0.0, F[:, :, t]) # F_t = Z_t * P_t * Z_t
    sum_matrix(F, H, t, 0) # F[:, :, t] .+= H
    return
end

function update_P_Ztransp_Finv(P_Ztransp_invF::AbstractArray{T}, ZP::AbstractArray{T}, F::AbstractArray{T}, t::Int) where T <: AbstractFloat
    LinearAlgebra.BLAS.gemm!('T', 'N', 1.0, ZP, invF(F, t), 0.0, P_Ztransp_invF)
    return
end