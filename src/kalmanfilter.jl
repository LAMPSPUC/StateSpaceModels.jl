"""Kalman Filter for state space time series systems with big Kappa initialization"""
function sqrt_kalmanfilter(sys::StateSpaceSystem, dim::StateSpaceDimensions, sqrtH::Array{Float64}, sqrtQ::Array{Float64}; tol = 1e-5)

    # Load dimension data
    n = dim.n
    p = dim.p
    m = dim.m
    r = dim.r

    # Load system data
    y = sys.y
    Z = sys.Z
    T = sys.T
    R = sys.R

    # Initial state: big Kappa initialization
    bigkappa = 1e6
    a0 = zeros(m)
    P0 = bigkappa*ones(m)
    sqrtP0 = diagm(P0)

    # One step ahead state predictor
    a = Array{Array}(n+1)
    # One step ahead sqrt variance predictor
    sqrtP = Array{Array}(n+1)
    # Innovation
    v = Array{Array}(n)
    # Sqrt variance of the innovation
    sqrtF = Array{Array}(n)
    # Kalman gain
    K = Array{Array}(n)
    # Steady state Kalman gain
    Ksteady = Array{Float64}(m, p)
    # Steady state variance
    Ps = Array{Float64}(m, m)
    # Auxiliary matrices
    U = zeros(p + m, m + p + r)
    U2star = Array{Array}(n)
    # Flag to indicate steady state
    steadystate = false
    # Instant in which state state was attained
    tsteady = n+1

    # Initial state
    a[1] = a0
    sqrtP[1] = sqrtP0
    sqrtPsteady = zeros(0, 0)

    # Kalman Filter
    for t = 1:n
        v[t] = y[t, :] - Z[t]*a[t]
        if steadystate == true
            a[t+1] = T*a[t] + Ksteady*v[t]
        else
            # QR decomposition of auxiliary matrix U to obtain U2star
            U = [Z[t]*sqrtP[t] sqrtH zeros(p, r); T*sqrtP[t] zeros(m, p) R*sqrtQ]
            G = qr(U')[1]
            Ustar = U*G
            U2star[t] = Ustar[(p + 1):(p + m), 1:p]
            sqrtF[t] = Ustar[1:p, 1:p]

            # Kalman gain
            K[t] = U2star[t]*pinv(sqrtF[t])
            a[t+1] = T*a[t] + K[t]*v[t]
            sqrtP[t+1] = Ustar[(p + 1):(p + m), (p + 1):(p + m)]

            # Verifying if steady state was attained
            if maximum(abs.((sqrtP[t+1] - sqrtP[t])/sqrtP[t+1])) < tol
                steadystate = true
                tsteady = t-1
                Ksteady = K[t-1]
                sqrtPsteady = sqrtP[t-1]
                a[t+1] = T*a[t] + Ksteady*v[t]
            end
        end
    end

    # Filter structure
    ss_filter = FilterOutput(a, v, steadystate, tsteady, Ksteady, U2star, sqrtP, sqrtF, sqrtPsteady)

    return ss_filter
end

"""Square-root smoother for state space time series systems"""
function sqrt_smoother(sys::StateSpaceSystem, dim::StateSpaceDimensions, ss_filter::FilterOutput)

    # Load dimension data
    n = dim.n
    m = dim.m
    p = dim.p
    p_exp = dim.p_exp

    # Load system data
    Z = sys.Z
    T = sys.T

    # Load filter data
    a = ss_filter.a
    v = ss_filter.v
    tsteady = ss_filter.tsteady
    Ksteady = ss_filter.Ksteady
    U2star = ss_filter.U2star
    sqrtF = ss_filter.sqrtF
    sqrtP = ss_filter.sqrtP
    sqrtPsteady = ss_filter.sqrtPsteady

    # Smoothed state
    alpha = Array{Array}(n)
    # Smoothed state variance
    V = Array{Array}(n)
    L = Array{Array}(n)
    r = Array{Array}(n)
    sqrtN = Array{Array}(n)

    # Initialization
    sqrtN[end] = zeros(m, m)
    r[end] = zeros(m)

    # Iterating backwards
    for t = n:-1:tsteady
        L[t] = T - Ksteady*Z[t]
        r[t-1] = Z[t]' * pinv(sqrtF[tsteady]*sqrtF[tsteady]') * v[t] + L[t]'*r[t]

        # QR decomposition of auxiliary matrix Nstar
        Nstar = [Z[t]' * pinv(sqrtF[tsteady]) L[t]'*sqrtN[t]]
        G = qr(Nstar')[1]
        NstarG = Nstar*G
        sqrtN[t-1] = NstarG[1:m, 1:m]

        # Computing smoothed state and its variance
        alpha[t] = a[t] + (sqrtPsteady*sqrtPsteady') * r[t-1]
        V[t] = (sqrtPsteady*sqrtPsteady') - (sqrtPsteady*sqrtPsteady')*(sqrtN[t]*sqrtN[t]')*(sqrtPsteady*sqrtPsteady')
    end

    for t = tsteady-1:-1:2
        L[t] = T - U2star[t] * pinv(sqrtF[t]) * Z[t]
        r[t-1] = Z[t]' * pinv(sqrtF[t]*sqrtF[t]') * v[t] + L[t]'*r[t]
        Nstar = [Z[t]'*pinv(sqrtF[t]) L[t]'*sqrtN[t]]

        # QR decomposition of auxiliary matrix Nstar
        G = qr(Nstar')[1]
        NstarG = Nstar*G
        sqrtN[t-1] = NstarG[1:m, 1:m]

        # Computing smoothed state and its variance
        alpha[t] = a[t] + (sqrtP[t]*sqrtP[t]')*r[t-1]
        V[t] = (sqrtP[t]*sqrtP[t]') - (sqrtP[t]*sqrtP[t]')*(sqrtN[t]*sqrtN[t]')*(sqrtP[t]*sqrtP[t]')
    end

    L[1] = T - U2star[1] * pinv(sqrtF[1]) * Z[1]
    r_0 = Z[1]' * pinv(sqrtF[1] * sqrtF[1]') * v[1] + L[1]' * r[1]
    Nstar = [Z[1]'*pinv(sqrtF[1]) L[1]'*sqrtN[1]]
    G = qr(Nstar')[1]
    NstarG = Nstar*G

    sqrtN_0 = NstarG[1:m, 1:m]
    alpha[1] = a[1] + (sqrtP[1]*sqrtP[1]') * r_0
    V[1] = (sqrtP[1]*sqrtP[1]') - (sqrtP[1]*sqrtP[1]')*(sqrtN_0*sqrtN_0')*(sqrtP[1]*sqrtP[1]')

    # Defining states
    trend = Array{Float64}(n, p)
    slope = Array{Float64}(n, p)
    seasonal = Array{Float64}(n, p)
    for i = 1:p
        for t = 1:n
            trend[t, i] = alpha[t][p_exp + i]
            slope[t, i] = alpha[t][p_exp + p + i]
            seasonal[t, i] = alpha[t][p_exp + 2*p + i]
        end
    end

    # State structure
    smoothedstate = SmoothedState(trend, slope, seasonal, V, alpha)

    return smoothedstate
end
