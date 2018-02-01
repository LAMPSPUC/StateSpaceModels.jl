using Optim, Distributions

mutable struct StateSpaceSystem
  y::Array # observations
  X::Array # exogenous variables
  n::Int # number of observations
  p::Int # number of endogenous variables
  m::Int # number of parameters
  r::Int # total number of states
  s::Int # seasonality
  Z::Array # observation matrices
  T::Array # state update matrix
  R::Array # state noise matrix
  a::Array
  sqrtP::Array
  v::Array
  sqrtF::Array
  K::Array
  Ks::Array
  sqrtPSteady::Array
  ts::Int
  alpha::Array # state
  V::Array
  psi::Array # parameters
  level::Array
  slope::Array
  seasonal::Array
  U2star::Array
  sqrtH::Any
  sqrtQ::Any
  steadyState::Bool
end

"""Create an empty system structure"""
function statespace_system()
    system = StateSpaceSystem(Array{Float64}(0), Array{Float64}(0), 0, 0, 0, 0, 0, Array{Float64}(0), Array{Float64}(0),
                    Array{Float64}(0), Array{Float64}(0), Array{Float64}(0), Array{Float64}(0), Array{Float64}(0),
                    Array{Float64}(0), Array{Float64}(0), Array{Float64}(0), 0, Array{Float64}(0), Array{Float64}(0),
                    Array{Float64}(0), Array{Float64}(0), Array{Float64}(0), Array{Float64}(0), Array{Float64}(0),
                    Array{Float64}(0), Array{Float64}(0), false)
    return system
end

"""Compute and return the log-likelihood concerning vector psitilde"""
function statespace_likelihood(psitilde::Array{Float64}, system::StateSpaceSystem)

    # Measurement covariance matrix
    if system.p > 1
        sqrtH = tril(ones(system.p, system.p))
        unknownsH = Int(system.p*(system.p + 1)/2)
        sqrtH[sqrtH .== 1] = psitilde[1:unknownsH]
    else
        sqrtH = [psitilde[1]]
        unknownsH = 1
    end

    # State covariance matrix
    sqrtQ = kron(eye(Int(system.r/system.p)), tril(ones(system.p, system.p)))
    covunknowns = Int(unknownsH + (system.r/system.p) * (system.p*(system.p + 1)/2))
    sqrtQ[sqrtQ .== 1] = psitilde[unknownsH+1:covunknowns]

    # Obtain the innovation vector v and its covariance matrix F
    system = sqrt_kalmanfilter(sqrtH, sqrtQ, system)

    # Compute the log-likelihood based on v and F
    if system.ts < system.n
        for cont = system.ts+1:system.n
            system.sqrtF[cont] = system.sqrtF[system.ts]
        end
    end
    loglikelihood = system.n*system.p*log(2*pi)/2

    for t = system.r:system.n
        loglikelihood = loglikelihood + .5 * (log(det(system.sqrtF[t]*system.sqrtF[t]')) + system.v[t]'*((system.sqrtF[t]*system.sqrtF[t]') \ system.v[t]))
    end

    return loglikelihood
end

"""Kalman Filter for time series state space models with big Kappa initialization"""
function sqrt_kalmanfilter(sqrtH::Array{Float64}, sqrtQ::Array{Float64}, system::StateSpaceSystem)

    # initial state
    bigKappa = 1000.0
    a0 = zeros(system.m)
    P0 = bigKappa*ones(system.m)
    sqrtP0 = diagm(P0)

    # one-step ahead state predictor
    a = Array{Array}(system.n+1)
    # one-step ahead square-root state variance predictor
    sqrtP = Array{Array}(system.n+1)
    # innovation
    v = Array{Array}(system.n)
    # square-root innovation variance
    sqrtF = Array{Array}(system.n)
    # Kalman gain
    K = Array{Array}(system.n)
    # steady state Kalman gain
    Ks = NaN*ones(system.m, system.p)
    # steady state variance
    Ps = NaN*ones(system.m, system.m)
    # partitioned auxiliary matrix for instant t
    U = zeros(system.p + system.m, system.m + system.p + system.r)
    U2star = Array{Array}(system.n)

    # flag to keep track if the system is in a steady state
    steadyState = false
    # time index that the system reached steady state
    ts = system.n+1
    # relative error tolerance
    tol = 1e-5
    # initial state
    a[1] = a0
    sqrtP[1] = sqrtP0
    sqrtPSteady = zeros(0,0)

    for t = 1:system.n
        v[t] = system.y[t] - system.Z[t]*a[t]
        if steadyState == true
            a[t+1] = system.T*a[t] + Ks*v[t]
        else
            U = [system.Z[t]*sqrtP[t] sqrtH zeros(system.p, system.r);
                 system.T*sqrtP[t] zeros(system.m, system.p) system.R*sqrtQ]
            G = qrfact!(U')[:Q]
            Ustar = U*G
            U2star[t] = Ustar[(system.p + 1):(system.p + system.m), 1:system.p]

            sqrtF[t] = Ustar[1:system.p, 1:system.p]
            K[t] = (sqrtF[t]' \ U2star[t]')'
            a[t+1] = system.T*a[t] + K[t]*v[t]
            sqrtP[t+1] = Ustar[(system.p + 1):(system.p + system.m), (system.p + 1):(system.p + system.m)]

            # verify if system has reached steady state
            if maximum(abs.((sqrtP[t+1] - sqrtP[t])/sqrtP[t+1])) < tol
                steadyState = true
                ts = t-1
                Ks = K[t-1]
                sqrtPSteady = sqrtP[t-1]
                a[t+1] = system.T*a[t] + Ks*v[t]
                system.steadyState = true
            end
        end
    end

    # Save
    system.a = a
    system.U2star = U2star
    system.sqrtP = sqrtP
    system.v = v
    system.sqrtF = sqrtF
    system.K = K
    system.ts = ts
    if ts < system.n+1
        system.Ks = Ks
        system.sqrtPSteady = sqrtPSteady
    end

    return system
end

"""Fixed interval smoother for time series state space models"""
function sqrt_smoother(system::StateSpaceSystem)

    # Load system
    Z = system.Z
    T = system.T
    n = system.n
    m = system.m
    ts = system.ts
    U2star = system.U2star
    sqrtF = system.sqrtF
    v = system.v
    a = system.a
    sqrtP = system.sqrtP

    if ts < n+1
        Ks = system.Ks
        sqrtPSteady = system.sqrtPSteady
    end

    # smoothed state
    alpha = Array{Array}(n)
    # smoothed variance
    V = Array{Array}(n)
    L = Array{Array}(n)
    # weighted sum of innovations after time (t-1)
    r = Array{Array}(n)
    # variance of the weighted sum of innovations after time (t-1)
    sqrtN = Array{Array}(n)

    # Initialization
    sqrtN[end] = zeros(m, m)
    r[end] = zeros(m)

    for t = n:-1:ts
        L[t] = T - Ks*Z[t]
        r[t-1] = Z[t]'*(sqrtF[ts]*sqrtF[ts]')^(-1)*v[t] + L[t]'*r[t]
        Nstar = [Z[t]'*sqrtF[ts]^(-1) L[t]'*sqrtN[t]]
        G, = qr(Nstar')
        NstarG = Nstar*G
        sqrtN[t-1] = NstarG[1:m, 1:m]
        # smoothed state recursion
        alpha[t] = a[t] + (sqrtPSteady * sqrtPSteady') * r[t-1]
        # smoothed variance recursion
        V[t] = (sqrtPSteady*sqrtPSteady') - (sqrtPSteady*sqrtPSteady')*(sqrtN[t]*sqrtN[t]')*(sqrtPSteady*sqrtPSteady')
    end

    t = 0
    for t = ts-1:-1:2
        L[t] = T - U2star[t] * (sqrtF[t]^(-1)) * Z[t]
        r[t-1] = Z[t]' * ((sqrtF[t]*sqrtF[t]')^(-1)) * v[t] + L[t]'*r[t]
        Nstar = [Z[t]'*sqrtF[t]^(-1) L[t]'*sqrtN[t]]
        G, = qr(Nstar')
        NstarG = Nstar*G
        sqrtN[t-1] = NstarG[1:m, 1:m]
        # smoothed state recursion
        alpha[t] = a[t] + (sqrtP[t]*sqrtP[t]')*r[t-1]
        # smoothed variance recursion
        V[t] = (sqrtP[t]*sqrtP[t]') - (sqrtP[t]*sqrtP[t]')*(sqrtN[t]*sqrtN[t]')*(sqrtP[t]*sqrtP[t]')
    end

    L[1] = T - U2star[1] * sqrtF[1]^(-1) * Z[1]
    r_0 = Z[1]' * (sqrtF[1] * sqrtF[1]')^(-1) * v[1] + L[1]' * r[1]
    Nstar = [Z[1]'*sqrtF[1]^(-1) L[1]'*sqrtN[1]]
    G, = qr(Nstar')
    NstarG = Nstar*G
    sqrtN0 = NstarG[1:m, 1:m]
    # smoothed state recursion
    alpha[1] = a[1] + (sqrtP[1]*sqrtP[1]')*r_0
    # smoothed variance recursion
    V[1] = (sqrtP[1]*sqrtP[1]') - (sqrtP[1]*sqrtP[1]')*(sqrtN0*sqrtN0')*(sqrtP[1]*sqrtP[1]')

    # save in system structure
    system.alpha = alpha
    system.V = V

    return system
end

"""Fits a basic structural model to y, returning the smoothed estimates and their variances"""
function statespace(y::Array{Float64}, s::Int; X = Array{Float64}(0,0), nseeds = 1)

    # create system structure
    system = statespace_system()
    system.X = X
    system.y = y
    system.s = s

    # number of endogenous observations and variables
    system.n, system.p = size(y[:,:])

    # number of exogenous observations and variables
    n_obs_exp, n_exp = size(X[:,:])

    if n_obs_exp > 0 && n_obs_exp != system.n
        error("Number of observations in X and y mismatch.")
    end

    info("Creating state space model with $(system.p) endogenous variable(s) and $n_exp exogenous variable(s).")

    system.m = (1 + system.s + n_exp)*system.p
    system.r = 3*system.p

    # measurement equation
    system.Z = Array{Array}(system.n)
    if n_exp > 0
        for t = 1:system.n
            system.Z[t] = kron([X[t, :]' 1 0 1 zeros(1, system.s - 2)], eye(system.p, system.p))
        end
    else
        for t = 1:system.n
            system.Z[t] = kron([1 0 1 zeros(1, system.s - 2)], eye(system.p, system.p))
        end
    end

    # state equation
    system.T = Array{Float64}(system.m, system.m)
    system.R = Array{Float64}(system.m, system.r)
    if n_exp > 0
        T0 = [eye(n_exp, n_exp) zeros(n_exp, 1 + system.s)]
        system.T = kron([T0; zeros(1, n_exp) 1 1 zeros(1, system.s - 1); zeros(1, n_exp) 0 1 zeros(1, system.s - 1);
            zeros(1, n_exp) 0 0 -ones(1, system.s - 1);
            zeros(system.s - 2, n_exp) zeros(system.s - 2, 2) eye(system.s - 2, system.s - 2) zeros(system.s - 2, 1)],
            eye(system.p, system.p))
    else
        system.T = kron([zeros(1, n_exp) 1 1 zeros(1, system.s - 1); zeros(1, n_exp) 0 1 zeros(1, system.s - 1);
            zeros(1, n_exp) 0 0 -ones(1, system.s - 1);
            zeros(system.s - 2, n_exp) zeros(system.s - 2, 2) eye(system.s - 2, system.s - 2) zeros(system.s - 2, 1)],
            eye(system.p, system.p))
    end
    system.R = kron([zeros(n_exp, 3); eye(3, 3); zeros(system.s - 2, 3)], eye(system.p, system.p))

    ## Maximum likelihood estimation
    # number of unknown parameters
    npsi = Int((1 + system.r/system.p)*(system.p*(system.p + 1)/2))
    # vector of unknowns
    bestpsi = Array{Float64}(npsi)
    # search limits
    inflim = -1e5
    suplim = 1e5

    # optimization
    info("Starting maximum likelihood estimation.")
    info("Running with $(length(procs())) procs in parallel.")

    seeds = SharedArray{Float64}(npsi, nseeds)
    bestpsi = SharedArray{Float64}(npsi)
    loglikelihood = SharedArray{Float64}(nseeds)
    psi = SharedArray{Float64}(npsi, nseeds)

    for iseed = 1:nseeds
        # uniformly generate values on the interval [inflim, suplim]
        seeds[:, iseed] = inflim + rand(inflim:suplim, npsi)
    end

    @sync @parallel for iseed = 1:nseeds
        # perform optimization
        info("Computing MLE for seed $iseed of $nseeds...")
        tryOpt = optimize(psitilde -> statespace_likelihood(psitilde, system), seeds[:, iseed],
                            LBFGS(), Optim.Options(f_tol = 1e-12, g_tol = 1e-12, iterations = 10^5,
                            show_trace = false))
        loglikelihood[iseed] = tryOpt.minimum
        psi[:, iseed] = tryOpt.minimizer
        info("Log-likelihood for seed $iseed: $(-loglikelihood[iseed])")
    end
    bestpsi = psi[:, indmin(loglikelihood)]
    system.psi = bestpsi

    info("Maximum likelihood estimation complete.")
    info("Log-Likelihood: $(maximum(-loglikelihood))")

    # Measurement's covariance matrix
    if system.p > 1
        sqrtH = tril(ones(system.p, system.p));
        unknownsH = Int(system.p * (system.p + 1) / 2)
        sqrtH[sqrtH .== 1] = bestpsi[1:unknownsH]
    else
        sqrtH = [bestpsi[1]]
        unknownsH = 1
    end
    # State covariance matrix
    sqrtQ = kron(eye(Int(system.r/system.p)), tril(ones(system.p, system.p)))
    sqrtQ[sqrtQ .== 1] = bestpsi[(unknownsH+1):Int(unknownsH + (system.r/system.p)*(system.p*(system.p + 1)/2))]

    ## Filter & Smoother
    system = sqrt_kalmanfilter(sqrtH, sqrtQ, system)
    system = sqrt_smoother(system)

    system.sqrtH = sqrtH
    system.sqrtQ = sqrtQ

    alphaMatrix = Array{Float64}(system.n, system.m)
    for t = 1:system.n, i = 1:system.m
        alphaMatrix[t,i] = system.alpha[t][i]
    end
    y = system.y
    v = system.v
    system.level = Array{Array}(system.p)
    system.slope = Array{Array}(system.p)
    system.seasonal = Array{Array}(system.p)

    for i = 1:system.p
        system.level[i] = alphaMatrix[:, n_exp + i]
        system.slope[i] = alphaMatrix[:, n_exp + system.p + i]
        system.seasonal[i] = alphaMatrix[:, n_exp + 2*system.p + i]
    end

    info("End of structural model estimation.")

    return system
end

"""Simulate S future scenarios up to N steps ahead from system y.
Output is a NxS matrix with each line representing an instant and each column representing a scenario."""
function simulate_statespace(system::StateSpaceSystem, N::Int, S::Int; X = Array{Float64}(0,0))

    # check if there are explanatory variables
    n_obs_exp, n_exp = size(X[:,:])

    if n_obs_exp > 0 && N > n_obs_exp
        info("Simulation period is larger than explanatory variables input")
    end

    Z = Array{Array}(N)
    if n_exp > 0
        for t = 1:N
            Z[t] = kron([X[t, :]' 1 0 1 zeros(1, system.s - 2)], eye(system.p, system.p))
        end
    else
        for t = 1:N
            Z[t] = kron([1 0 1 zeros(1, system.s - 2)], eye(system.p, system.p))
        end
    end

    # observation innovation
    dist_ϵ = MvNormal(zeros(system.p), system.sqrtH*system.sqrtH')

    # state innovation
    dist_η = MvNormal(zeros(system.r), system.sqrtQ*system.sqrtQ')

    αsim = Array{Float64}(N,system.m)
    ysim = Array{Array}(S)

    for s = 1:S
        ysim[s] = Array{Float64}(N, system.p)
        ϵ = rand(dist_ϵ, N)'
        η = rand(dist_η, N)'

        for t = 1:N
            if t == 1
                αsim[t,:] = system.T*system.alpha[system.n] + (system.R*η[t,:])
                ysim[s][t,:] = Z[t]*αsim[t,:] + ϵ[t,:]
            else
                αsim[t,:] = system.T*αsim[t-1,:] + (system.R*η[t,:])
                ysim[s][t,:] = Z[t]*αsim[t,:] + ϵ[t,:]
            end
        end
    end

    scenarios = Array{Float64}(N, S)
    for t = 1:N, s = 1:S
        scenarios[t,s] = ysim[s][t]
    end

    return scenarios
end
