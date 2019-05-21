"""
    sqrt_kalmanfilter(sys::StateSpaceSystem, sqrtH::Array{Float64}, sqrtQ::Array{Float64}; tol::Float64 = 1e-5)

Square-root Kalman filter with big Kappa initialization.
"""
function sqrt_kalmanfilter(sys::StateSpaceSystem, sqrtH::Array{Float64}, sqrtQ::Array{Float64}; tol::Float64 = 1e-5)

    # Load dimensions
    n = sys.dim.n
    p = sys.dim.p
    m = sys.dim.m
    r = sys.dim.r

    # Load system
    y = sys.y
    Z = sys.Z
    T = sys.T
    R = sys.R

    # Initial state: big Kappa initialization
    bigkappa = 1e6
    a0       = zeros(m, 1)
    P0       = bigkappa * ones(m)
    sqrtP0   = Matrix(Diagonal(P0))

    # Predictive state and its sqrt-covariance
    a     = Vector{Matrix{Float64}}(undef, n+1)
    sqrtP = Vector{Matrix{Float64}}(undef, n+1)

    # Innovation and its sqrt-covariance
    v     = Vector{Matrix{Float64}}(undef, n)
    sqrtF = Vector{Matrix{Float64}}(undef, n)

    # Kalman gain and steady-state Kalman gain
    K       = Vector{Matrix{Float64}}(undef, n)
    Ksteady = Matrix{Float64}(undef, m, p)

    # Predictive variance in steady state
    Ps = Matrix{Float64}(undef, m, m)

    # Auxiliary matrices
    U      = zeros(p + m, m + p + r)
    U2star = Vector{Matrix{Float64}}(undef, n)
    
    # Steady state flags
    steadystate = false
    tsteady     = n+1

    # Initial state
    a[1]        = a0
    sqrtP[1]    = sqrtP0
    sqrtPsteady = zeros(0, 0)

    # Square-root Kalman filter
    for t = 1:n
        v[t] = y[t, :] - Z[t]*a[t]
        if steadystate == true
            a[t+1] = T * a[t] + Ksteady * v[t]
        else
            # QR decomposition of auxiliary matrix U to obtain Ustar
            U            = [Z[t]*sqrtP[t] sqrtH zeros(p, r); T*sqrtP[t] zeros(m, p) R*sqrtQ]
            G            = qr(U').Q
            Ustar        = U*G
            U2star[t]    = Ustar[(p + 1):(p + m), 1:p]
            sqrtF[t]     = Ustar[1:p, 1:p]

            # Kalman gain
            K[t]       = U2star[t] * pinv(sqrtF[t])
            a[t+1]     = T * a[t] + K[t] * v[t]
            sqrtP[t+1] = Ustar[(p + 1):(p + m), (p + 1):(p + m)]

            # Checking if steady state was attained
            if maximum(abs.((sqrtP[t+1] - sqrtP[t])/sqrtP[t+1])) < tol
                steadystate = true
                tsteady     = t-1
                Ksteady     = K[t-1]
                sqrtPsteady = sqrtP[t-1]
                a[t+1]      = T*a[t] + Ksteady*v[t]
            end
        end
    end

    # Saving in filter structure
    ss_filter = FilterOutput(a, v, steadystate, tsteady, Ksteady, U2star, sqrtP, sqrtF, sqrtPsteady)

    return ss_filter
end

"""
    sqrt_smoother(sys::StateSpaceSystem, ss_filter::FilterOutput)

Square-root smoother for state space model.
"""
function sqrt_smoother(sys::StateSpaceSystem, ss_filter::FilterOutput)

    # Load dimensions data
    n = sys.dim.n
    m = sys.dim.m
    p = sys.dim.p

    # Load system data
    Z = sys.Z
    T = sys.T

    # Load filter data
    a           = ss_filter.a
    v           = ss_filter.v
    tsteady     = ss_filter.tsteady
    Ksteady     = ss_filter.Ksteady
    U2star      = ss_filter.U2star
    sqrtF       = ss_filter.sqrtF
    sqrtP       = ss_filter.sqrtP
    sqrtPsteady = ss_filter.sqrtPsteady

    # Smoothed state and its covariance
    alpha = Vector{Matrix{Float64}}(undef, n)
    V     = Vector{Matrix{Float64}}(undef, n)
    L     = Vector{Matrix{Float64}}(undef, n)
    r     = Vector{Matrix{Float64}}(undef, n)
    sqrtN = Vector{Matrix{Float64}}(undef, n)

    # Initialization
    sqrtN[end] = zeros(m, m)
    r[end]     = zeros(m, 1)

    # Iterating backwards
    for t = n:-1:tsteady
        L[t]   = T - Ksteady*Z[t]
        r[t-1] = Z[t]' * pinv(sqrtF[tsteady] * sqrtF[tsteady]') * v[t] + L[t]' * r[t]

        # QR decomposition of auxiliary matrix Nstar
        Nstar        = [Z[t]' * pinv(sqrtF[tsteady]) L[t]' * sqrtN[t]]
        G            = qr(Nstar').Q
        NstarG       = Nstar * G
        sqrtN[t-1]   = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        alpha[t] = a[t] + (sqrtPsteady * sqrtPsteady') * r[t-1]
        V[t]     = (sqrtPsteady * sqrtPsteady') - 
                   (sqrtPsteady * sqrtPsteady') *
                   (sqrtN[t] * sqrtN[t]') * 
                   (sqrtPsteady * sqrtPsteady')
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
