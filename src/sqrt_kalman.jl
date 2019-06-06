
"""
    sqrt_kalmanfilter(model::StateSpaceModel, sqrtH::Matrix{Typ}, sqrtQ::Matrix{Typ}; tol::Float64 = 1e-5) where Typ <: AbstractFloat

Square-root Kalman filter with big Kappa initialization.
"""
function sqrt_kalmanfilter(model::StateSpaceModel, aux::AuxiliarySqrtKalman, sqrtH::Matrix{Typ}, sqrtQ::Matrix{Typ}; tol::Float64 = 1e-5) where Typ <: AbstractFloat

    # Load dimensions
    n, p, m, r = size(model)

    # Load system
    y = model.y
    Z, T, R = ztr(model)
    
    # Steady state initialization
    steadystate  = false
    tsteady      = n+1

    # Initial state: big Kappa initialization
    aux.a[1, :]        = zeros(m, 1)
    aux.sqrtP[:, :, 1]    = 1e6 .* Matrix(I, m, m)

    # Pre-allocating for performance (check if it is memory efficient)
    aux.sqrtH_zeros_pr  .= [sqrtH aux.zeros_pr]
    aux.zeros_mp_RsqrtQ .= [aux.zeros_mp R*sqrtQ]

    # Square-root Kalman filter
    for t = 1:n
        aux.v[t, :] = y[t, :] - Z[:, :, t] * aux.a[t, :]

        if steadystate
            aux.sqrtF[:, :, t] = aux.sqrtF[:, :, t-1]
            aux.K[:, :, t] = aux.K[:, :, t-1]
            aux.a[t+1, :] = T * aux.a[t, :] + aux.K[:, :, t] * aux.v[t, :]
            # No need to keep storing
            aux.sqrtP[:, :, t+1] = aux.sqrtP[:, :, t]

        else
            # Manipulation of auxiliary matrices
            # bla = [Z[:, :, t] * aux.sqrtP[:, :, t] aux.sqrtH_zeros_pr]
            # ble = [T * aux.sqrtP[:, :, t] aux.zeros_mp_RsqrtQ]
            aux.U .= [Z[:, :, t] * aux.sqrtP[:, :, t] aux.sqrtH_zeros_pr; T * aux.sqrtP[:, :, t] aux.zeros_mp_RsqrtQ]
            aux.G .= aux.U'
            # Inplace QR factorization
            bla= qr(aux.G).Q
            aux.U *= bla
            aux.U2star[:, :, t] = aux.U[aux.range1, aux.range2]
            aux.sqrtF[:, :, t]  = aux.U[aux.range2, aux.range2]

            # Kalman gain and predictive state update
            # Note: sqrtF[:, :, t] is lower triangular
            aux.K[:, :, t] = aux.U2star[:, :, t] * pinv(aux.sqrtF[:, :, t])
            # forward_substitution()
            aux.a[t+1, :]        = T * aux.a[t, :] + aux.K[:, :, t] * aux.v[t, :]
            aux.sqrtP[:, :, t+1] = aux.U[aux.range1, aux.range1]

            # Checking if steady state was attained
            if maximum(abs.((aux.sqrtP[:, :, t+1] - aux.sqrtP[:, :, t]) / aux.sqrtP[:, :, t+1])) < tol
                steadystate = true
                tsteady     = t
            end
        end
    end

    # Saving in filter structure
    kfilter = FilterOutput(aux.a[1:end-1, :], aux.v, aux.sqrtP, aux.sqrtF, steadystate, tsteady)

    return kfilter, aux.U2star, aux.K
end

"""
    sqrt_smoother(model::StateSpaceModel, kfilter::FilterOutput)

Square-root smoother for state space model.
"""
function sqrt_smoother(model::StateSpaceModel, kfilter::FilterOutput, U2star::Array{Float64, 3}, K::Array{Float64, 3})

    # Load dimensions data
    n, p, m, r = size(model)

    # Load system data
    Z, T, R = ztr(model)

    # Load filter data
    a           = kfilter.a
    v           = kfilter.v
    tsteady     = kfilter.tsteady
    sqrtF       = kfilter.sqrtF
    sqrtP       = kfilter.sqrtP

    # Smoothed state and its covariance
    alpha = Matrix{Float64}(undef, n, m)
    V     = Array{Float64, 3}(undef, m, m, n)
    L     = Array{Float64, 3}(undef, m, m, n)
    r     = Matrix{Float64}(undef, n, m)
    sqrtN = Array{Float64, 3}(undef, m, m, n)

    # Initialization
    sqrtN[:, :, end]  = zeros(m, m)
    r[end, :]      = zeros(m, 1)
    sqrtPsteady = sqrtP[:, :, end]
    sqrtFsteady = sqrtF[:, :, end]

    # Iterating backwards
    for t = n:-1:tsteady
        L[:, :, t]   = T - K[:, :, end]*Z[:, :, t]
        r[t-1, :] = Z[:, :, t]' * pinv(sqrtFsteady*sqrtFsteady') * v[t, :] + L[:, :, t]' * r[t, :]

        # QR decomposition of auxiliary matrix Nstar
        Nstar        = [Z[:, :, t]' * pinv(sqrtFsteady) L[:, :, t]' * sqrtN[:, :, t]]
        G            = qr(Nstar').Q
        NstarG       = Nstar * G
        sqrtN[:, :, t-1]   = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        alpha[t, :] = a[t, :] + (sqrtPsteady*sqrtPsteady') * r[t-1, :]
        V[:, :, t]     = (sqrtPsteady*sqrtPsteady') - 
                   (sqrtPsteady*sqrtPsteady') *
                   (sqrtN[:, :, t]*sqrtN[:, :, t]') * 
                   (sqrtPsteady*sqrtPsteady')
    end

    for t = tsteady-1:-1:2
        L[:, :, t]   = T - U2star[:, :, t] * pinv(sqrtF[:, :, t]) * Z[:, :, t]
        r[t-1, :] = Z[:, :, t]' * pinv(sqrtF[:, :, t] * sqrtF[:, :, t]') * v[t, :] + L[:, :, t]'*r[t, :]
        Nstar  = [Z[:, :, t]' * pinv(sqrtF[:, :, t]) L[:, :, t]' * sqrtN[:, :, t]]

        # QR decomposition of auxiliary matrix Nstar
        G          = qr(Nstar').Q
        NstarG     = Nstar*G
        sqrtN[:, :, t-1] = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        alpha[t, :] = a[t, :] + (sqrtP[:, :, t] * sqrtP[:, :, t]') * r[t-1, :]
        V[:, :, t]     = (sqrtP[:, :, t] * sqrtP[:, :, t]') - 
                   (sqrtP[:, :, t] * sqrtP[:, :, t]') *
                   (sqrtN[:, :, t] * sqrtN[:, :, t]') *
                   (sqrtP[:, :, t] * sqrtP[:, :, t]')
    end

    L[:, :, 1]   = T - U2star[:, :, 1] * pinv(sqrtF[:, :, 1]) * Z[:, :, 1]
    r_0    = Z[:, :, 1]' * pinv(sqrtF[:, :, 1] * sqrtF[:, :, 1]') * v[1, :] + L[:, :, 1]' * r[1, :]
    Nstar  = [Z[:, :, 1]' * pinv(sqrtF[:, :, 1]) L[:, :, 1]' * sqrtN[:, :, 1]]
    G      = qr(Nstar').Q
    NstarG = Nstar*G

    sqrtN_0  = NstarG[1:m, 1:m]
    alpha[1, :] = a[1, :] + (sqrtP[:, :, 1]*sqrtP[:, :, 1]') * r_0
    V[:, :, 1]  = (sqrtP[:, :, 1] * sqrtP[:, :, 1]') - 
                  (sqrtP[:, :, 1] * sqrtP[:, :, 1]') * 
                  (sqrtN_0 * sqrtN_0') *
                  (sqrtP[:, :, 1] * sqrtP[:, :, 1]')

    # Smoothed state structure
    smoothedstate = SmoothedState(alpha, V)

    return smoothedstate
end