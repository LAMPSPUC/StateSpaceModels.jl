"""
    sqrt_kalmanfilter(model::StateSpaceModel, sqrtH::Matrix{Typ}, sqrtQ::Matrix{Typ}; tol::Float64 = 1e-5) where Typ <: AbstractFloat

Square-root Kalman filter with big Kappa initialization.
"""
function sqrt_kalmanfilter(model::StateSpaceModel, sqrtH::Matrix{Typ}, sqrtQ::Matrix{Typ}; tol::Float64 = 1e-5) where Typ <: AbstractFloat

    # Load dimensions
    n, p, m, r = size(model)

    # Load system
    y = model.y
    Z, T, R = ztr(model)

    # Predictive state and its sqrt-covariance
    a     = Vector{Matrix{Float64}}(undef, n+1)
    sqrtP = Vector{Matrix{Float64}}(undef, n+1)

    # Innovation and its sqrt-covariance
    v     = Vector{Matrix{Float64}}(undef, n)
    sqrtF = Vector{Matrix{Float64}}(undef, n)

    # Kalman gain
    K       = Vector{Matrix{Float64}}(undef, n)

    # Auxiliary matrices
    U2star = Vector{Matrix{Float64}}(undef, n)
    
    # Steady state initialization
    steadystate  = false
    tsteady      = n+1

    # Initial state: big Kappa initialization
    a[1]        = zeros(m, 1)
    sqrtP[1]    = 1e6.*Matrix(I, m, m)

    # Pre-allocating for performance
    zeros_pr = zeros(p, r)
    zeros_mp = zeros(m, p)
    range1   = (p + 1):(p + m)
    range2   = 1:p
    sqrtH_zeros_pr  = [sqrtH zeros_pr]
    zeros_mp_RsqrtQ = [zeros_mp R*sqrtQ]

    # Square-root Kalman filter
    for t = 1:n
        v[t] = y[t, :] - Z[t]*a[t]
        if steadystate
            sqrtF[t] = sqrtF[t-1]
            K[t] = K[t-1]
            a[t+1] = T * a[t] + K[t] * v[t]
            sqrtP[t+1] = sqrtP[t]
        else
            # Manipulation of auxiliary matrices
            U         = SMatrix{p + m, m + p + r}([Z[t]*sqrtP[t] sqrtH_zeros_pr; T*sqrtP[t] zeros_mp_RsqrtQ])
            G         = qr(U').Q
            Ustar     = U*G
            U2star[t] = Ustar[range1, range2]
            sqrtF[t]  = Ustar[range2, range2]

            # Kalman gain and predictive state update
            K[t]       = U2star[t]*pinv(sqrtF[t])
            a[t+1]     = T*a[t] + K[t]*v[t]
            sqrtP[t+1] = Ustar[range1, range1]

            # Checking if steady state was attained
            if maximum(abs.((sqrtP[t+1] - sqrtP[t])/sqrtP[t+1])) < tol
                steadystate = true
                tsteady     = t
            end
        end
    end

    # Saving in filter structure
    kfilter = FilterOutput(a, v, sqrtP, sqrtF, steadystate, tsteady)

    return kfilter, U2star, K
end

"""
    sqrt_smoother(model::StateSpaceModel, kfilter::FilterOutput)

Square-root smoother for state space model.
"""
function sqrt_smoother(model::StateSpaceModel, kfilter::FilterOutput, U2star::Vector{Matrix{Float64}}, K::Vector{Matrix{Float64}})

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
    alpha = Vector{Matrix{Float64}}(undef, n)
    V     = Vector{Matrix{Float64}}(undef, n)
    L     = Vector{Matrix{Float64}}(undef, n)
    r     = Vector{Matrix{Float64}}(undef, n)
    sqrtN = Vector{Matrix{Float64}}(undef, n)

    # Initialization
    sqrtN[end]  = zeros(m, m)
    r[end]      = zeros(m, 1)
    sqrtPsteady = sqrtP[end]
    sqrtFsteady = sqrtF[end]

    # Iterating backwards
    for t = n:-1:tsteady
        L[t]   = T - K[end]*Z[t]
        r[t-1] = Z[t]' * pinv(sqrtFsteady*sqrtFsteady') * v[t] + L[t]' * r[t]

        # QR decomposition of auxiliary matrix Nstar
        Nstar        = [Z[t]' * pinv(sqrtFsteady) L[t]' * sqrtN[t]]
        G            = qr(Nstar').Q
        NstarG       = Nstar * G
        sqrtN[t-1]   = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        alpha[t] = a[t] + (sqrtPsteady*sqrtPsteady') * r[t-1]
        V[t]     = (sqrtPsteady*sqrtPsteady') - 
                   (sqrtPsteady*sqrtPsteady') *
                   (sqrtN[t]*sqrtN[t]') * 
                   (sqrtPsteady*sqrtPsteady')
    end

    for t = tsteady-1:-1:2
        L[t]   = T - U2star[t] * pinv(sqrtF[t]) * Z[t]
        r[t-1] = Z[t]' * pinv(sqrtF[t] * sqrtF[t]') * v[t] + L[t]'*r[t]
        Nstar  = [Z[t]' * pinv(sqrtF[t]) L[t]' * sqrtN[t]]

        # QR decomposition of auxiliary matrix Nstar
        G          = qr(Nstar').Q
        NstarG     = Nstar*G
        sqrtN[t-1] = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        alpha[t] = a[t] + (sqrtP[t] * sqrtP[t]') * r[t-1]
        V[t]     = (sqrtP[t] * sqrtP[t]') - 
                   (sqrtP[t] * sqrtP[t]') *
                   (sqrtN[t] * sqrtN[t]') *
                   (sqrtP[t] * sqrtP[t]')
    end

    L[1]   = T - U2star[1] * pinv(sqrtF[1]) * Z[1]
    r_0    = Z[1]' * pinv(sqrtF[1] * sqrtF[1]') * v[1] + L[1]' * r[1]
    Nstar  = [Z[1]' * pinv(sqrtF[1]) L[1]' * sqrtN[1]]
    G      = qr(Nstar').Q
    NstarG = Nstar*G

    sqrtN_0  = NstarG[1:m, 1:m]
    alpha[1] = a[1] + (sqrtP[1]*sqrtP[1]') * r_0
    V[1]     = (sqrtP[1] * sqrtP[1]') - 
               (sqrtP[1] * sqrtP[1]') * 
               (sqrtN_0 * sqrtN_0') *
               (sqrtP[1] * sqrtP[1]')

    # Smoothed state structure
    smoothedstate = SmoothedState(alpha, V)

    return smoothedstate
end