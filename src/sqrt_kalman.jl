
"""
    sqrt_kalmanfilter(model::StateSpaceModel, sqrtH::Matrix{Typ}, sqrtQ::Matrix{Typ}; tol::Float64 = 1e-5) where Typ <: AbstractFloat

Square-root Kalman filter with big Kappa initialization.
"""
function sqrt_kalmanfilter!(model::StateSpaceModel, filter_data::FilterOutput, aux_data::AuxiliaryDataSqrtKalman, sqrtH::Matrix{Typ}, sqrtQ::Matrix{Typ}; tol::Float64 = 1e-5) where Typ <: AbstractFloat

    # Load dimensions
    n, p, m, r = size(model)

    # Load system
    y       = model.y
    Z, T, R = ztr(model)

    # Load filter data
    a, v         = filter_data.a, filter_data.v
    sqrtF, sqrtP = filter_data.sqrtF, filter_data.sqrtP

    # Load smoother auxiliary prealocated variables
    sqrtH_zeros_pr  = aux_data.sqrtH_zeros_pr
    zeros_pr        = aux_data.zeros_pr
    zeros_mp_RsqrtQ = aux_data.zeros_mp_RsqrtQ
    zeros_mp        = aux_data.zeros_mp
    K, G            = aux_data.K, aux_data.G
    U, U2star       = aux_data.U, aux_data.U2star
    range1, range2  = aux_data.range1, aux_data.range2
    
    # Steady state initialization
    steadystate  = false
    tsteady      = n + 1

    # Initial state: big Kappa initialization
    a[1, :]        .= zeros(m)
    sqrtP[:, :, 1] .= 1e6 .* Matrix(I, m, m)

    # Pre-allocating for performance (check if it is memory efficient)
    sqrtH_zeros_pr  .= [sqrtH zeros_pr]
    zeros_mp_RsqrtQ .= [zeros_mp R * sqrtQ]

    # Square-root Kalman filter
    for t = 1:n
        v[t, :] = y[t, :] - Z[:, :, t] * a[t, :]

        if steadystate
            sqrtF[:, :, t] = sqrtF[:, :, t-1]
            K[:, :, t] = K[:, :, t-1]
            a[t+1, :] = T * a[t, :] + K[:, :, t] * v[t, :]
            # No need to keep storing
            sqrtP[:, :, t+1] = sqrtP[:, :, t]

        else
            # Manipulation of auxiliary matrices
            U .= [Z[:, :, t] * sqrtP[:, :, t] sqrtH_zeros_pr; T * sqrtP[:, :, t] zeros_mp_RsqrtQ]
            G .= U'
            # Inplace QR factorization
            bla = qr(G).Q
            U *= bla
            U2star[:, :, t] .= U[range1, range2]
            sqrtF[:, :, t]  .= U[range2, range2]

            # Kalman gain and predictive state update
            # Note: sqrtF[:, :, t] is lower triangular
            K[:, :, t] .= U2star[:, :, t] * pinv(sqrtF[:, :, t])
            # forward_substitution()
            a[t+1, :]        .= T * a[t, :] + K[:, :, t] * v[t, :]
            sqrtP[:, :, t+1] .= U[range1, range1]

            # Checking if steady state was attained
            if maximum(abs.((sqrtP[:, :, t+1] - sqrtP[:, :, t]) / sqrtP[:, :, t+1])) < tol
                steadystate = true
                tsteady     = t
            end
        end
    end

    # Saving steady state info
    filter_data.steadystate, filter_data.tsteady = steadystate, tsteady

    return nothing
end

"""
    sqrt_smoother(model::StateSpaceModel, kfilter::FilterOutput)

Square-root smoother for state space model.
"""
function sqrt_smoother!(model::StateSpaceModel, filter_data::FilterOutput, smoother_data::SmoothedState, aux_data::AuxiliaryDataSqrtKalman)

    # Load dimensions data
    n, p, m, r = size(model)

    # Load system data
    Z, T, R = ztr(model)

    # Load filter data
    a, v         = filter_data.a, filter_data.v
    sqrtF, sqrtP = filter_data.sqrtF, filter_data.sqrtP
    tsteady      = filter_data.tsteady

    # Load smoother prealocated variables
    alpha, V = smoother_data.alpha, smoother_data.V

    # Load smoother auxiliary prealocated variables
    U2star      = aux_data.U2star
    K           = aux_data.K
    r, L        = aux_data.r, aux_data.L
    sqrtN       = aux_data.sqrtN
    sqrtPsteady = aux_data.sqrtPsteady
    sqrtFsteady = aux_data.sqrtFsteady

    # Initialization
    sqrtPsteady = sqrtP[:, :, end]
    sqrtFsteady = sqrtF[:, :, end]

    # Iterating backwards
    for t = n:-1:tsteady
        L[:, :, t]   = T - K[:, :, end] * Z[:, :, t]
        r[t-1, :] = Z[:, :, t]' * pinv(sqrtFsteady*sqrtFsteady') * v[t, :] + L[:, :, t]' * r[t, :]

        # QR decomposition of auxiliary matrix Nstar
        Nstar        = [Z[:, :, t]' * pinv(sqrtFsteady) L[:, :, t]' * sqrtN[:, :, t]]
        G            = qr(Nstar').Q
        NstarG       = Nstar * G
        sqrtN[:, :, t-1]   = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        alpha[t, :] = a[t, :] + (sqrtPsteady*sqrtPsteady') * r[t-1, :]
        V[:, :, t]  = sqrtPsteady * sqrtPsteady' - sqrtPsteady * sqrtPsteady' * sqrtN[:, :, t] * sqrtN[:, :, t]' * sqrtPsteady * sqrtPsteady'
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
    smoother_data.alpha .= alpha
    smoother_data.V     .= V

end