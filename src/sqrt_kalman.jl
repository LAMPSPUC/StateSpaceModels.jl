"""
    sqrt_kalman_filter(model::StateSpaceModel{Typ}, sqrtH::Matrix{Typ}, sqrtQ::Matrix{Typ}; tol::Typ = 1e-5) where Typ

Square-root Kalman filter with big Kappa initialization.
"""
function sqrt_kalman_filter(model::StateSpaceModel{Typ}, sqrtH::Matrix{Typ}, sqrtQ::Matrix{Typ}; tol::Typ = Typ(1e-5)) where Typ

    time_invariant = model.mode == "time-invariant"

    # Load dimensions
    n, p, m, r = size(model)

    # Load system
    y = model.y
    Z, T, R = ztr(model)

    # Predictive state and its sqrt-covariance
    a     = Matrix{Typ}(undef, n+1, m)
    sqrtP = Array{Typ, 3}(undef, m, m, n+1)

    # Innovation and its sqrt-covariance
    v     = Matrix{Typ}(undef, n, p)
    sqrtF = Array{Typ, 3}(undef, p, p, n)

    # Kalman gain
    K     = Array{Typ, 3}(undef, m, p, n)

    # Auxiliary matrices
    U2star = Array{Typ, 3}(undef, m, p, n)

    # Steady state initialization
    steadystate  = false
    tsteady      = n+1

    # Initial state: big Kappa initialization
    a[1, :]        = zeros(m, 1)
    sqrtP[:, :, 1]    = 1e6 .* Matrix(I, m, m)

    # Pre-allocating for performance
    zeros_pr = zeros(p, r)
    zeros_mp = zeros(m, p)
    range1   = (p + 1):(p + m)
    range2   = 1:p
    sqrtH_zeros_pr  = [sqrtH zeros_pr]
    zeros_mp_RsqrtQ = [zeros_mp R*sqrtQ]

    !isempty(model.missing_observations) && error("Treatment of missing values not implemented for SquareRootFilter, please use KalmanFilter.")

    # Square-root Kalman filter
    for t = 1:n
        v[t, :] = y[t, :] - Z[:, :, t]*a[t, :]
        if steadystate
            sqrtF[:, :, t] = sqrtF[:, :, t-1]
            K[:, :, t] = K[:, :, t-1]
            a[t+1, :] = T * a[t, :] + K[:, :, t] * v[t, :]
            sqrtP[:, :, t+1] = sqrtP[:, :, t]
        else
            # Manipulation of auxiliary matrices
            U         = SMatrix{p + m, m + p + r}([Z[:, :, t]*sqrtP[:, :, t] sqrtH_zeros_pr;
                                                   T*sqrtP[:, :, t]         zeros_mp_RsqrtQ])
            G         = qr(U').Q
            Ustar     = U*G
            U2star[:, :, t] = Ustar[range1, range2]
            sqrtF[:, :, t]  = Ustar[range2, range2]

            # Kalman gain and predictive state update
            K[:, :, t]       = U2star[:, :, t]*inv(sqrtF[:, :, t])
            a[t+1, :]        = T*a[t, :] + K[:, :, t]*v[t, :]
            sqrtP[:, :, t+1] = Ustar[range1, range1]

            # Checking if steady state was attained
            if time_invariant && check_steady_state(sqrtP, t, tol)
                steadystate = true
                tsteady     = t
            end
        end
    end

    # Return the auxiliary filter structre
    return SquareRootFilter(a, v, sqrtP, sqrtF, steadystate, tsteady, K)
end

"""
    filtered_state(model::StateSpaceModel{Typ}, sqrt_filter::SquareRootFilter{Typ}) where Typ

Obtain the filtered state estimates and their covariance matrices.
"""
function filtered_state(model::StateSpaceModel{Typ}, sqrt_filter::SquareRootFilter{Typ}) where Typ

    # Load dimensions data
    n, p, m, r = size(model)

    # Load system data
    Z, T, R = ztr(model)

    # Load filter data
    a = sqrt_filter.a
    v = sqrt_filter.v
    F = gram_in_time(sqrt_filter.sqrtF)
    P = gram_in_time(sqrt_filter.sqrtP)

    # Filtered state and its covariance
    att = Matrix{Typ}(undef, n, m)
    Ptt = Array{Typ, 3}(undef, m, m, n)

    for t = 1:n
        PZF = P[:, :, t] * Z[:, :, t]' * inv(F[:, :, t])
        att[t, :]    = a[t, :] + PZF * v[t, :]
        Ptt[:, :, t] = P[:, :, t] - PZF * Z[:, :, t] * P[:, :, t]
        ensure_pos_sym!(Ptt, t)
    end

    return att, Ptt

end

"""
    sqrt_smoother(model::StateSpaceModel{Typ}, sqrt_filter::SquareRootFilter{Typ}) where Typ

Square-root smoother for state space model.
"""
function sqrt_smoother(model::StateSpaceModel{Typ}, sqrt_filter::SquareRootFilter{Typ}) where Typ

    # Load dimensions data
    n, p, m, r = size(model)

    # Load system data
    Z, T, R = ztr(model)

    # Load filter data
    a           = sqrt_filter.a
    v           = sqrt_filter.v
    tsteady     = sqrt_filter.tsteady
    sqrtF       = sqrt_filter.sqrtF
    sqrtP       = sqrt_filter.sqrtP
    K           = sqrt_filter.K

    # Smoothed state and its covariance
    alpha = Matrix{Typ}(undef, n, m)
    V     = Array{Typ, 3}(undef, m, m, n)
    L     = Array{Typ, 3}(undef, m, m, n)
    r     = Matrix{Typ}(undef, n, m)
    sqrtN = Array{Typ, 3}(undef, m, m, n)

    # Initialization
    sqrtN[:, :, end]  = zeros(m, m)
    r[end, :]      = zeros(m, 1)
    sqrtPsteady = sqrtP[:, :, end]
    sqrtFsteady = sqrtF[:, :, end]

    # Iterating backwards
    for t = n:-1:tsteady
        Psteady = gram(sqrtP[:, :, end])
        Fsteady = gram(sqrtF[:, :, end])
        N_t_1   = gram(sqrtN[:, :, t-1])
        L[:, :, t]   = T - K[:, :, end] * Z[:, :, t]
        r[t-1, :] = Z[:, :, t]' * inv(Fsteady) * v[t, :] + L[:, :, t]' * r[t, :]

        # QR decomposition of auxiliary matrix Nstar
        Nstar        = [Z[:, :, t]' * inv(sqrtF[:, :, end]) L[:, :, t]' * sqrtN[:, :, t]]
        G            = qr(Nstar').Q
        NstarG       = Nstar * G
        sqrtN[:, :, t-1]   = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        alpha[t, :] = a[t, :] + Psteady * r[t-1, :]
        V[:, :, t]  = Psteady - Psteady * N_t_1 * Psteady
    end

    for t = tsteady-1:-1:2
        L[:, :, t]   = T - K[:, :, t] * Z[:, :, t]
        r[t-1, :] = Z[:, :, t]' * inv(sqrtF[:, :, t] * sqrtF[:, :, t]') * v[t, :] + L[:, :, t]'*r[t, :]
        Nstar  = [Z[:, :, t]' * inv(sqrtF[:, :, t]) L[:, :, t]' * sqrtN[:, :, t]]

        # QR decomposition of auxiliary matrix Nstar
        NstarG     = Nstar*qr(Nstar').Q
        sqrtN[:, :, t-1] = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        P = gram(sqrtP[:, :, t])
        N = gram(sqrtN[:, :, t-1])
        alpha[t, :] = a[t, :] + P * r[t-1, :]
        V[:, :, t]  = P - (P * N * P)
    end

    L[:, :, 1]   = T - K[:, :, 1] * Z[:, :, 1]
    r_0    = Z[:, :, 1]' * inv(gram(sqrtF[:, :, 1])) * v[1, :] + L[:, :, 1]' * r[1, :]
    Nstar  = [Z[:, :, 1]' * inv(sqrtF[:, :, 1]) L[:, :, 1]' * sqrtN[:, :, 1]]
    G      = qr(Nstar').Q
    NstarG = Nstar*G

    sqrtN_0  = NstarG[1:m, 1:m]
    P_1 = gram(sqrtP[:, :, 1])
    alpha[1, :] = a[1, :] + P_1 * r_0
    V[:, :, 1]  = P_1 - (P_1 * gram(sqrtN_0) * P_1)

    # Return the Square Root kalman filter smoothed state
    return Smoother(alpha, V)
end


# All filters have to have implemented the following functions
# *
# *
# *

function statespace_covariance(psi::Vector{T}, p::Int, r::Int,
                               filter_type::Type{SquareRootFilter{T}}) where T
    # Observation sqrt-covariance matrix
    if p > 1
        sqrtH     = tril!(ones(p, p))
        unknownsH = Int(p*(p + 1)/2)
        sqrtH[findall(isequal(1), sqrtH)] = psi[1:unknownsH]
    else
        sqrtH = psi[1].*ones(1, 1)
        unknownsH = 1
    end

    # State sqrt-covariance matrix
    sqrtQ = kron(Matrix{T}(I, Int(r/p), Int(r/p)), tril!(ones(p, p)))
    sqrtQ[findall(x -> x == 1, sqrtQ)] = psi[(unknownsH+1):Int(unknownsH + (r/p)*(p*(p + 1)/2))]

    return T.(sqrtH), T.(sqrtQ)
end

function get_log_likelihood_params(psitilde::Vector{T}, model::StateSpaceModel,
                                   filter_type::Type{SquareRootFilter{T}}) where T

    sqrtH, sqrtQ = statespace_covariance(psitilde, model.dim.p, model.dim.r, filter_type)
    # Obtain innovation v and its variance F
    sqrt_sqrt_filter = sqrt_kalman_filter(model, sqrtH, sqrtQ)
    # Return v and F
    return sqrt_sqrt_filter.v, gram_in_time(sqrt_sqrt_filter.sqrtF)
end

function kfas(model::StateSpaceModel{T}, covariance::StateSpaceCovariance{T},
              filter_type::Type{SquareRootFilter{T}}) where T
    # Compute sqrt matrices
    sqrtH = cholesky(covariance.H).L # .L stands for Lower triangular
    sqrtQ = cholesky(covariance.Q).L # .L stands for Lower triangular

    # Do the SquareRootFilter
    filter_output  = sqrt_kalman_filter(model, sqrtH.data, sqrtQ.data)
    smoothed_state = sqrt_smoother(model, filter_output)
    att, Ptt       = filtered_state(model, filter_output)

    return FilterOutput(filter_output.a, att, filter_output.v,
                         gram_in_time(filter_output.sqrtP), Ptt, gram_in_time(filter_output.sqrtF),
                         filter_output.steadystate, filter_output.tsteady) ,
           SmoothedState(smoothed_state.alpha, smoothed_state.V)
end
