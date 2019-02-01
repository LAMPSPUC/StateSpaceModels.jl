"""
    sqrt_kalmanfilter(sys::StateSpaceSystem, dim::StateSpaceDimensions, sqrtH::Array{Float64}, sqrtQ::Array{Float64}; tol::Float64 = 1e-5)

Square-root Kalman filter for time series with big Kappa initialization.
"""
function sqrt_kalmanfilter(sys::StateSpaceSystem, dim::StateSpaceDimensions, sqrtH::Array{Float64}, sqrtQ::Array{Float64}; tol::Float64 = 1e-5)

    # Load dimensions data
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
    sqrtP0 = Matrix(Diagonal(P0))

    # Predictive state and its sqrt-covariance
    a = Array{Array}(undef, n+1)
    sqrtP = Array{Array}(undef, n+1)
    # Innovation and its sqrt-covariance
    v = Array{Array}(undef, n)
    sqrtF = Array{Array}(undef, n)
    # Kalman gain and steady-state Kalman gain
    K = Array{Array}(undef, n)
    Ksteady = Array{Float64}(undef, m, p)
    # Predictive variance in steady state
    Ps = Array{Float64}(undef, m, m)
    # Auxiliary matrices
    U = zeros(p + m, m + p + r)
    U2star = Array{Array}(undef, n)
    # Steady state flags
    steadystate = false
    tsteady = n+1

    # Initial state
    a[1] = a0
    sqrtP[1] = sqrtP0
    sqrtPsteady = zeros(0, 0)

    # Square-root Kalman filter
    for t = 1:n
        v[t] = y[t, :] - Z[t]*a[t]
        if steadystate == true
            a[t+1] = T*a[t] + Ksteady*v[t]
        else
            # QR decomposition of auxiliary matrix U to obtain Ustar
            U = [Z[t]*sqrtP[t] sqrtH zeros(p, r); T*sqrtP[t] zeros(m, p) R*sqrtQ]
            G = LinearAlgebra.qr(U').Q
            Ustar = U*G
            U2star[t] = Ustar[(p + 1):(p + m), 1:p]
            sqrtF[t] = Ustar[1:p, 1:p]

            # Kalman gain
            K[t] = U2star[t]*LinearAlgebra.pinv(sqrtF[t])
            a[t+1] = T*a[t] + K[t]*v[t]
            sqrtP[t+1] = Ustar[(p + 1):(p + m), (p + 1):(p + m)]

            # Checking if steady state was attained
            if maximum(abs.((sqrtP[t+1] - sqrtP[t])/sqrtP[t+1])) < tol
                steadystate = true
                tsteady = t-1
                Ksteady = K[t-1]
                sqrtPsteady = sqrtP[t-1]
                a[t+1] = T*a[t] + Ksteady*v[t]
            end
        end
    end

    # Saving in filter structure
    ss_filter = FilterOutput(a, v, steadystate, tsteady, Ksteady, U2star, sqrtP, sqrtF, sqrtPsteady)

    return ss_filter
end

"""
    sqrt_smoother(sys::StateSpaceSystem, dim::StateSpaceDimensions, ss_filter::FilterOutput)

Square-root smoothing for time series state space model.
"""
function sqrt_smoother(sys::StateSpaceSystem, dim::StateSpaceDimensions, ss_filter::FilterOutput)

    # Load dimensions data
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

    # Smoothed state and its covariance
    alpha = Array{Array}(undef, n)
    V = Array{Array}(undef, n)
    L = Array{Array}(undef, n)
    r = Array{Array}(undef, n)
    sqrtN = Array{Array}(undef, n)

    # Initialization
    sqrtN[end] = zeros(m, m)
    r[end] = zeros(m)

    # Iterating backwards
    for t = n:-1:tsteady
        L[t] = T - Ksteady*Z[t]
        r[t-1] = Z[t]' * LinearAlgebra.pinv(sqrtF[tsteady]*sqrtF[tsteady]') * v[t] + L[t]'*r[t]

        # QR decomposition of auxiliary matrix Nstar
        Nstar = [Z[t]' * LinearAlgebra.pinv(sqrtF[tsteady]) L[t]'*sqrtN[t]]
        G = LinearAlgebra.qr(Nstar').Q
        NstarG = Nstar*G
        sqrtN[t-1] = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        alpha[t] = a[t] + (sqrtPsteady*sqrtPsteady') * r[t-1]
        V[t] = (sqrtPsteady*sqrtPsteady') - (sqrtPsteady*sqrtPsteady')*(sqrtN[t]*sqrtN[t]')*(sqrtPsteady*sqrtPsteady')
    end

    for t = tsteady-1:-1:2
        L[t] = T - U2star[t] * LinearAlgebra.pinv(sqrtF[t]) * Z[t]
        r[t-1] = Z[t]' * LinearAlgebra.pinv(sqrtF[t]*sqrtF[t]') * v[t] + L[t]'*r[t]
        Nstar = [Z[t]'*LinearAlgebra.pinv(sqrtF[t]) L[t]'*sqrtN[t]]

        # QR decomposition of auxiliary matrix Nstar
        G = LinearAlgebra.qr(Nstar').Q
        NstarG = Nstar*G
        sqrtN[t-1] = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        alpha[t] = a[t] + (sqrtP[t]*sqrtP[t]')*r[t-1]
        V[t] = (sqrtP[t]*sqrtP[t]') - (sqrtP[t]*sqrtP[t]')*(sqrtN[t]*sqrtN[t]')*(sqrtP[t]*sqrtP[t]')
    end

    L[1] = T - U2star[1] * LinearAlgebra.pinv(sqrtF[1]) * Z[1]
    r_0 = Z[1]' * LinearAlgebra.pinv(sqrtF[1] * sqrtF[1]') * v[1] + L[1]' * r[1]
    Nstar = [Z[1]'*LinearAlgebra.pinv(sqrtF[1]) L[1]'*sqrtN[1]]
    G = LinearAlgebra.qr(Nstar').Q
    NstarG = Nstar*G

    sqrtN_0 = NstarG[1:m, 1:m]
    alpha[1] = a[1] + (sqrtP[1]*sqrtP[1]') * r_0
    V[1] = (sqrtP[1]*sqrtP[1]') - (sqrtP[1]*sqrtP[1]')*(sqrtN_0*sqrtN_0')*(sqrtP[1]*sqrtP[1]')

    # Defining smoothed state vectors
    trend = Array{Float64}(undef, n, p)
    slope = Array{Float64}(undef, n, p)
    seasonal = Array{Float64}(undef, n, p)
    exogenous = Array{Array}(undef, p_exp)
    for j = 1:p_exp
        exogenous[j] = zeros(n, p)
    end
    for i = 1:p
        for t = 1:n
            for j = 1:p_exp
                exogenous[j][t, i] = alpha[t][j-1 + i]*sys.X[t, j]
            end
            trend[t, i] = alpha[t][p_exp + i]
            slope[t, i] = alpha[t][p_exp + p + i]
            seasonal[t, i] = alpha[t][p_exp + 2*p + i]
        end
    end

    # Smoothed state structure
    smoothedstate = SmoothedState(trend, slope, seasonal, exogenous, V, alpha)

    return smoothedstate
end
