using Optim, Distributions

"""State space system dimensions"""
struct StateSpaceDimensions
    n::Int
    p::Int
    m::Int
    r::Int
    p_exp::Int
end

"""State space data matrices"""
struct StateSpaceSystem
    y::Array # observations
    X::Array # exogenous variables
    s::Int # seasonality period
    Z::Array # observation matrix
    T::Array # state matrix
    R::Array # state noise matrix
end

"""Fixed parameters of the state space system"""
mutable struct StateSpaceParameters
    sqrtH::Array
    sqrtQ::Array
end

"""Smoothed state data"""
struct SmoothedState
    trend::Array # trend
    slope::Array # slope
    seasonal::Array # seasonal
    V::Array # state variance
    alpha::Array # state matrix
end

"""Kalman Filter output"""
mutable struct FilterOutput
    a::Array
    v::Array
    steadystate::Bool
    tsteady::Int
    Ksteady::Array
    U2star::Array
    sqrtP::Array
    sqrtF::Array
    sqrtPsteady::Array
end

"""Output data structure"""
struct StateSpace
    sys::StateSpaceSystem # system matrices
    dim::StateSpaceDimensions # system dimensions
    state::SmoothedState # smoothed state
    param::StateSpaceParameters # fixed parameters
    steadystate::Bool # flag that indicates if steady state was attained
end

"""Compute covariance matrix from given parameters"""
function statespace_covariance(psi::Array{Float64,1}, p::Int, r::Int)

    # Observation covariance matrix
    if p > 1
        sqrtH = tril(ones(p, p))
        unknownsH = Int(p * (p + 1) / 2)
        sqrtH[sqrtH .== 1] = psi[1:unknownsH]
    else
        sqrtH = [psi[1]]
        unknownsH = 1
    end

    # State covariance matrix
    sqrtQ = kron(eye(Int(r/p)), tril(ones(p, p)))
    sqrtQ[sqrtQ .== 1] = psi[(unknownsH+1):Int(unknownsH + (r/p)*(p*(p + 1)/2))]

    return sqrtH, sqrtQ
end

"""Compute and return the log-likelihood concerning the parameter vector psitilde"""
function statespace_likelihood(psitilde::Array{Float64,1}, sys::StateSpaceSystem, dim::StateSpaceDimensions)

    sqrtH, sqrtQ = statespace_covariance(psitilde, dim.p, dim.r)

    # Obtain innovation vector v and its covariance matrix F
    ss_filter = sqrt_kalmanfilter(sys, dim, sqrtH, sqrtQ)

    # Compute log-likelihood based on v and F
    if ss_filter.tsteady < dim.n
        for cont = ss_filter.tsteady+1:dim.n
            ss_filter.sqrtF[cont] = ss_filter.sqrtF[ss_filter.tsteady]
        end
    end
    loglikelihood = dim.n*dim.p*log(2*pi)/2

    for t = dim.m:dim.n
        loglikelihood = loglikelihood + .5 * (log(det(ss_filter.sqrtF[t]*ss_filter.sqrtF[t]')) + ss_filter.v[t]'*((ss_filter.sqrtF[t]*ss_filter.sqrtF[t]') \ ss_filter.v[t]))
    end

    return loglikelihood
end

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
        v[t] = y[t] - Z[t]*a[t]
        if steadystate == true
            a[t+1] = T*a[t] + Ksteady*v[t]
        else
            # QR decomposition of auxiliary matrix U to obtain U2star
            U = [Z[t]*sqrtP[t] sqrtH zeros(p, r); T*sqrtP[t] zeros(m, p) R*sqrtQ]
            G = qrfact!(U')[:Q]
            Ustar = U*G
            U2star[t] = Ustar[(p + 1):(p + m), 1:p]
            sqrtF[t] = Ustar[1:p, 1:p]

            # Kalman gain
            K[t] = (sqrtF[t]' \ U2star[t]')'
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
    alpha = Array{Float64}(n, m)
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
        r[t-1] = Z[t]'*(sqrtF[tsteady]*sqrtF[tsteady]')^(-1) * v[t] + L[t]'*r[t]

        # QR decomposition of auxiliary matrix Nstar
        Nstar = [Z[t]'*sqrtF[tsteady]^(-1) L[t]'*sqrtN[t]]
        G, = qr(Nstar')
        NstarG = Nstar*G
        sqrtN[t-1] = NstarG[1:m, 1:m]

        # Computing smoothed state and its variance
        alpha[t, :] = a[t] + (sqrtPsteady * sqrtPsteady') * r[t-1]
        V[t] = (sqrtPsteady*sqrtPsteady') - (sqrtPsteady*sqrtPsteady')*(sqrtN[t]*sqrtN[t]')*(sqrtPsteady*sqrtPsteady')
    end

    for t = tsteady-1:-1:2
        L[t] = T - U2star[t]*(sqrtF[t]^(-1))*Z[t]
        r[t-1] = Z[t]'*((sqrtF[t]*sqrtF[t]')^(-1)) * v[t] + L[t]'*r[t]
        Nstar = [Z[t]'*sqrtF[t]^(-1) L[t]'*sqrtN[t]]

        # QR decomposition of auxiliary matrix Nstar
        G, = qr(Nstar')
        NstarG = Nstar*G
        sqrtN[t-1] = NstarG[1:m, 1:m]

        # Computing smoothed state and its variance
        alpha[t, :] = a[t] + (sqrtP[t]*sqrtP[t]')*r[t-1]
        V[t] = (sqrtP[t]*sqrtP[t]') - (sqrtP[t]*sqrtP[t]')*(sqrtN[t]*sqrtN[t]')*(sqrtP[t]*sqrtP[t]')
    end

    L[1] = T - U2star[1] * sqrtF[1]^(-1) * Z[1]
    r_0 = Z[1]' * (sqrtF[1] * sqrtF[1]')^(-1) * v[1] + L[1]' * r[1]
    Nstar = [Z[1]'*sqrtF[1]^(-1) L[1]'*sqrtN[1]]
    G, = qr(Nstar')
    NstarG = Nstar*G

    sqrtN_0 = NstarG[1:m, 1:m]
    alpha[1, :] = a[1] + (sqrtP[1]*sqrtP[1]')*r_0
    V[1] = (sqrtP[1]*sqrtP[1]') - (sqrtP[1]*sqrtP[1]')*(sqrtN_0*sqrtN_0')*(sqrtP[1]*sqrtP[1]')

    # Defining states
    trend = Array{Float64}(n, p)
    slope = Array{Float64}(n, p)
    seasonal = Array{Float64}(n, p)
    for i = 1:p
        trend[:, i] = alpha[:, p_exp + i]
        slope[:, i] = alpha[:, p_exp + p + i]
        seasonal[:, i] = alpha[:, p_exp + 2*p + i]
    end

    # State structure
    smoothedstate = SmoothedState(trend, slope, seasonal, V, alpha)

    return smoothedstate
end

"""Estimate fixed parameters of a state space system"""
function estimate_statespace(sys::StateSpaceSystem, dim::StateSpaceDimensions, nseeds::Int;
    f_tol = 1e-12, g_tol = 1e-12, iterations = 10^5)

    # Initialization
    npsi = Int((1 + dim.r/dim.p)*(dim.p*(dim.p + 1)/2))
    seeds = SharedArray{Float64}(npsi, nseeds)
    loglikelihood = SharedArray{Float64}(nseeds)
    psi = SharedArray{Float64}(npsi, nseeds)

    # Initial conditions limits
    inflim = -1e5
    suplim = 1e5
    seedrange = collect(inflim:suplim)

    # Avoiding zero values for covariance
    deleteat!(seedrange, find(seedrange .== 0.0))

    info("Initiating maximum likelihood estimation with $nseeds seeds.")
    if length(procs()) > 1
        info("Running with $(length(procs())) procs in parallel.")
    else
        info("Running with $(length(procs())) proc.")
    end

    # Setting seed
    srand(123)

    # Generate random initial values in [inflim, suplim]
    for iseed = 1:nseeds
        seeds[:, iseed] = rand(seedrange, npsi)
    end

    # Optimization
    @sync @parallel for iseed = 1:nseeds
        info("Optimizing likelihood for seed $iseed of $nseeds...")
        optseed = optimize(psitilde -> statespace_likelihood(psitilde, sys, dim), seeds[:, iseed],
                            LBFGS(), Optim.Options(f_tol = f_tol, g_tol = g_tol, iterations = iterations,
                            show_trace = false))
        loglikelihood[iseed] = -optseed.minimum
        psi[:, iseed] = optseed.minimizer
        info("Log-likelihood for seed $iseed: $(loglikelihood[iseed])")
    end

    info("Maximum likelihood estimation complete.")
    info("Log-likelihood: $(maximum(loglikelihood))")

    bestpsi = psi[:, indmax(loglikelihood)]
    sqrtH, sqrtQ = statespace_covariance(bestpsi, dim.p, dim.r)

    # Parameter structure
    param = StateSpaceParameters(sqrtH, sqrtQ)

    return param
end

function statespace(y::Array{Float64,1}, s::Int; X = Array{Float64,2}(0,0), nseeds = 1)
    n = length(y)
    y = reshape(y, (n,1))
    statespace(y, s; X = X, nseeds = nseeds)
end

"""Estimate basic structural model and obtain smoothed state"""
function statespace(y::Array{Float64,2}, s::Int; X = Array{Float64,2}(0,0), nseeds = 1)

    # Number of observations and endogenous variables
    n, p = size(y)

    # Number of observations and exogenous variables
    n_exp, p_exp = size(X)

    if p_exp > 0 && n_exp < n
        error("Number of observations in X e y are incompatible.")
    end

    info("Creating structural model with $p endogenous variables and $p_exp exogenous variables.")

    m = (1 + s + p_exp)*p
    r = 3*p

    # Observation equation
    Z = Array{Array}(n)
    for t = 1:n
        Z[t] = p_exp > 0 ? kron([X[t, :]' 1 0 1 zeros(1, s - 2)], eye(p, p)) :
                           kron([1 0 1 zeros(1, s - 2)], eye(p, p))
    end

    # State equation
    if p_exp > 0
        T0 = [eye(p_exp, p_exp) zeros(p_exp, 1 + s)]
        T = kron([T0; zeros(1, p_exp) 1 1 zeros(1, s - 1); zeros(1, p_exp) 0 1 zeros(1, s - 1);
            zeros(1, p_exp) 0 0 -ones(1, s - 1);
            zeros(s - 2, p_exp) zeros(s - 2, 2) eye(s - 2, s - 2) zeros(s - 2, 1)],
            eye(p, p))
    else
        T = kron([zeros(1, p_exp) 1 1 zeros(1, s - 1); zeros(1, p_exp) 0 1 zeros(1, s - 1);
            zeros(1, p_exp) 0 0 -ones(1, s - 1);
            zeros(s - 2, p_exp) zeros(s - 2, 2) eye(s - 2, s - 2) zeros(s - 2, 1)],
            eye(p, p))
    end
    R = kron([zeros(p_exp, 3); eye(3, 3); zeros(s - 2, 3)], eye(p, p))

    # Creating data structures
    dim = StateSpaceDimensions(n, p, m, r, p_exp)
    sys = StateSpaceSystem(y, X, s, Z, T, R)

    # Maximum likelihood estimation
    ss_par = estimate_statespace(sys, dim, nseeds)

    # Kalman Filter and smoothing
    ss_filter = sqrt_kalmanfilter(sys, dim, ss_par.sqrtH, ss_par.sqrtQ)
    smoothedstate = sqrt_smoother(sys, dim, ss_filter)

    info("End of structural model estimation.")

    output = StateSpace(sys, dim, smoothedstate, ss_par, ss_filter.steadystate)

    return output
end

"""Simulate S future scenarios up to N periods ahead.
    Returns an NxS matrix where each line represents a period and each column represents a scenario."""
function simulate_statespace(ss::StateSpace, N::Int, S::Int)

    # Number of observations and exogenous variables
    n_exp, p_exp = size(ss.sys.X)

    if p_exp > 0 && N > n_exp - ss.dim.n
        error("Simulation period is larger than number of observations of future exogenous variables.")
    end

    # Distribution of observation innovation
    dist_ϵ = MvNormal(zeros(ss.dim.p), ss.param.sqrtH*ss.param.sqrtH')
    # Distribution of state innovation
    dist_η = MvNormal(zeros(ss.dim.r), ss.param.sqrtQ*ss.param.sqrtQ')

    αsim = Array{Float64}(N, ss.dim.m)
    ysim = Array{Array}(S)

    for s = 1:S
        ysim[s] = Array{Float64}(N, ss.dim.p)

        # Simulating innovations
        ϵ = rand(dist_ϵ, N)'
        η = rand(dist_η, N)'

        # Generating scenarios from innovations
        for t = 1:N
            if t == 1
                αsim[t, :] = ss.sys.T*ss.state.alpha[ss.dim.n, :] + (ss.sys.R*η[t, :])
                ysim[s][t, :] = ss.sys.Z[t]*αsim[t, :] + ϵ[t, :]
            else
                αsim[t, :] = ss.sys.T*αsim[t-1, :] + (ss.sys.R*η[t, :])
                ysim[s][t, :] = ss.sys.Z[t]*αsim[t, :] + ϵ[t, :]
            end
        end
    end

    # Creating scenarios matrix
    scenarios = Array{Float64}(N, S)
    for t = 1:N, s = 1:S
        scenarios[t, s] = ysim[s][t]
    end

    return scenarios
end
