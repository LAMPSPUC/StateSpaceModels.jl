using Optim, Plots, Distributions

mutable struct System
  y::Array # observations
  X::Array # exogenous variables
  n::Int # number of observations
  p::Int # number of endogenous variables
  m::Int # number of parameters
  r::Int # ??
  s::Int # seasonality
  Z::Array # observation matrices
  T::Array # state update matrix
  R::Array # state noise matrix
  a::Array
  Psqrt::Array
  v::Array
  Fsqrt::Array
  K::Array
  Ks::Array
  PsqrtSteady::Array
  ts::Int
  alpha::Array
  V::Array
  psi::Array
  level::Array
  slope::Array
  seasonal::Array
  U2star::Array
  Hsqrt::Any
  Qsqrt::Any
  steadyState::Bool
end

function create_system()
    system = System(Array{Float64}(0), Array{Float64}(0), 0, 0, 0, 0, 0, Array{Float64}(0), Array{Float64}(0),
                    Array{Float64}(0), Array{Float64}(0), Array{Float64}(0), Array{Float64}(0), Array{Float64}(0),
                    Array{Float64}(0), Array{Float64}(0), Array{Float64}(0), 0, Array{Float64}(0), Array{Float64}(0),
                    Array{Float64}(0), Array{Float64}(0), Array{Float64}(0), Array{Float64}(0), Array{Float64}(0),
                    Array{Float64}(0), Array{Float64}(0), false)
    return system
end

function state_space_likelihood(psiTilde, system)
    ## DESCRIPTION
    #
    #   Compute and return the log-likelihood concerning vector psiTilde.
    # ------------------------------------------------------------------------------

    # Measurement's covariance matrix
    if system.p > 1
        Hsqrt = tril(ones(system.p, system.p))
        nUnknownsH = Int(system.p*(system.p + 1)/2)
        Hsqrt[Hsqrt .== 1] = psiTilde[1:nUnknownsH]
    else
        Hsqrt = psiTilde[1]
        nUnknownsH = 1
    end

    # State's covariance matrix
    Qsqrt = kron(eye(Int(system.r/system.p)), tril(ones(system.p, system.p)))
    covUnknowns = Int(nUnknownsH + (system.r/system.p) * (system.p*(system.p + 1)/2))
    Qsqrt[Qsqrt .== 1] = psiTilde[nUnknownsH + 1 : covUnknowns]

    ## Obtain the innovation vector v and its covariance matrix F
    system = square_root_kalman_filter(Hsqrt, Qsqrt, system)

    ## Compute the log-likelihood based on v and F
    if system.ts < system.n
        for cont = system.ts+1 : system.n
            system.Fsqrt[cont] = system.Fsqrt[system.ts]
        end
    end
    logLikelihood = system.n*system.p*log(2*pi)/2

    try
        for t = 2 : system.n
            logLikelihood = logLikelihood + .5 * (log(det(system.Fsqrt[t]*system.Fsqrt[t]')) + system.v[t]' * ((system.Fsqrt[t]*system.Fsqrt[t]') \ system.v[t]))
        end
    catch
        println(Fsqrt)
    end

    return logLikelihood
end

function square_root_kalman_filter(Hsqrt, Qsqrt, system)
    ## Kalman Filter for time-series state-space models of the form:
    #
    #   measurement equations:
    #   y(t) = Z * alpha(t) + epslon(t), epslon(t)~N(0,H)
    #   state equations:
    #   alpha(t+1) = T * alpha(t) + R * eta(t) , eta(t)~N(0,Q)
    #
    #  INPUTS:
    #
    #  OUTPUT:
    #
    # note: cell's index {.} correspond to the time index t.

    bigKappa = 1000.0

    # initial state
    a0 = zeros(system.m)
    P0 = bigKappa*ones(system.m)
    P0sqrt = diagm(P0)

    ## Memory Allocation
    # one-step ahead state predictor
    a = Array{Array}(system.n + 1)
    # one-step ahead square-root state's variance predictor
    Psqrt = Array{Array}(system.n + 1)
    # innovations
    v = Array{Array}(system.n)
    # square-root innovations variance
    Fsqrt = Array{Array}(system.n)
    # Kalman gain
    K = Array{Array}(system.n)
    # steady state Kalman gain
    Ks = NaN*ones(system.m, system.p)
    # steady state variance
    Ps = NaN*ones(system.m, system.m)
    # partitioned auxiliary matrix for the instant t
    U = zeros(system.p + system.m, system.m + system.p + system.r)
    # lower triangular partitioned auxiliary matrix
    # Ustar = zeros(system.p + system.m, system.m + system.p + system.r)
    # partions of Ustar
    # U1star = Array{Array}(system.n)
    U2star = Array{Array}(system.n)
    # U3star = Array{Array}(system.n)

    ## Initialization
    # flag to keep track if the system is in a steady state
    steadyState = false
    # time index that the system reached steady state
    ts = system.n + 1
    # error tolerance
    tol = 1e-5
    # initial state
    a[1] = a0
    Psqrt[1] = P0sqrt
    PsqrtSteady = zeros(0,0)

    ## Recursion
    for t = 1 : system.n
        v[t] = system.y[t] - system.Z[t]*a[t]
        if steadyState == true
            # updating step
            a[t+1] = system.T*a[t] + Ks*v[t]
        else
            # build the auxiliary matrix
            U = [system.Z[t]*Psqrt[t] Hsqrt zeros(system.p, system.r);
                 system.T*Psqrt[t] zeros(system.m, system.p) system.R*Qsqrt]
            # orthogonal-triangular decomposition
            G = qrfact!(U')[:Q]
            Ustar = U*G
            # partitions of Ustar
            # U1star[t] = Ustar[1:system.p, 1:system.p]
            U2star[t] = Ustar[(system.p + 1):(system.p + system.m), 1:system.p]
            # U3star[t] = Ustar[(system.p + 1):(system.p + system.m), (system.p + 1):(system.p + system.m)]
            # innovation's variance
            Fsqrt[t] = Ustar[1:system.p, 1:system.p]
            # Kalman gain
            K[t] = (Fsqrt[t]' \ U2star[t]')'
            # updating step
            a[t+1] = system.T*a[t] + K[t]*v[t]
            Psqrt[t+1] = Ustar[(system.p + 1):(system.p + system.m), (system.p + 1):(system.p + system.m)]
            # test if the system reached steady state
            if maximum(abs.(Psqrt[t+1] - Psqrt[t])) < tol
                steadyState = true
                ts = t - 1
                Ks = K[t-1]
                PsqrtSteady = Psqrt[t-1]
                a[t+1] = system.T*a[t] + Ks*v[t]
                system.steadyState = true
            end
        end
    end

    ## Save
    system.a = a
    system.U2star = U2star
    system.Psqrt = Psqrt
    system.v = v
    system.Fsqrt = Fsqrt
    system.K = K
    system.ts = ts
    if ts < system.n+1
        system.Ks = Ks
        system.PsqrtSteady = PsqrtSteady
    end

    return system
end

function square_root_smoother(system)
    ## Fixed-Interval Smoother for time-series state-space models of the form:
    #
    #   measurement equations:
    #   y(t) = Z * alpha(t) + epslon(t), epslon(t)~N(0,H)
    #   state equations:
    #   alpha(t+1) = T * alpha(t) + R * eta(t) , eta(t)~N(0,Q)
    #
    # note: cell's index {.} correspond to the time index t.

    ## Load system
    Z = system.Z
    T = system.T
    n = system.n
    m = system.m
    ts = system.ts
    U2star = system.U2star
    Fsqrt = system.Fsqrt
    v = system.v
    a = system.a
    Psqrt = system.Psqrt

    if ts < n+1
        Ks = system.Ks;
        PsqrtSteady = system.PsqrtSteady
    end

    ## Memory Allocation
    # smoothed state
    alpha = Array{Array}(n)
    # smoothed variance
    V = Array{Array}(n)
    #
    L = Array{Array}(n)
    # weighted sum of innovations after time (t-1)
    r = Array{Array}(n)
    # variance of the weighted sum of innovations after time (t-1)
    Nsqrt = Array{Array}(n)

    ## Initialization
    Nsqrt[end] = zeros(m, m)
    r[end] = zeros(m)

    ## Recursion
    for t = n : -1 : ts
        L[t] = T - Ks*Z[t]
        r[t-1] = Z[t]' * (Fsqrt[ts] * Fsqrt[ts]')^(-1) * v[t] + L[t]' * r[t]
        # auxiliary matrix
        Nstar = [Z[t]' * Fsqrt[ts]^(-1) L[t]' * Nsqrt[t]]
        # orthogonal-triangular decomposition
        G, = qr(Nstar')
        NstarG = Nstar * G
        Nsqrt[t-1] = NstarG[1:m, 1:m]
        # smoothed state recursion
        alpha[t] = a[t] + (PsqrtSteady * PsqrtSteady') * r[t-1]
        # smoothed variance recursion
        V[t] = (PsqrtSteady*PsqrtSteady') - (PsqrtSteady*PsqrtSteady')*(Nsqrt[t]*Nsqrt[t]')*(PsqrtSteady*PsqrtSteady')
    end

    t = 0
    for t = (ts-1) : -1 : 2
        L[t] = T - U2star[t] * (Fsqrt[t]^(-1)) * Z[t]
        r[t-1] = Z[t]' * ((Fsqrt[t]*Fsqrt[t]')^(-1)) * v[t] + L[t]'*r[t]
        # auxiliary matrix
        Nstar = [Z[t]'*Fsqrt[t]^(-1) L[t]'*Nsqrt[t]]
        # orthogonal-triangular decomposition
        G, = qr(Nstar')
        NstarG = Nstar*G
        Nsqrt[t-1] = NstarG[1:m, 1:m]
        # smoothed state recursion
        alpha[t] = a[t] + (Psqrt[t]*Psqrt[t]')*r[t-1]
        # smoothed variance recursion
        V[t] = (Psqrt[t]*Psqrt[t]') - (Psqrt[t]*Psqrt[t]')*(Nsqrt[t]*Nsqrt[t]')*(Psqrt[t]*Psqrt[t]')
    end

    L[1] = T - U2star[1] * Fsqrt[1]^(-1) * Z[1]
    r_0 = Z[1]' * (Fsqrt[1] * Fsqrt[1]')^(-1) * v[1] + L[1]' * r[1]
    # auxiliary matrix
    Nstar = [Z[1]'*Fsqrt[1]^(-1) L[1]'*Nsqrt[1]]
    # orthogonal-triangular decomposition
    G, = qr(Nstar')
    NstarG = Nstar*G
    Nsqrt_0 = NstarG[1:m, 1:m]
    # smoothed state recursion
    alpha[1] = a[1] + (Psqrt[1]*Psqrt[1]')*r_0
    # smoothed variance recursion
    V[1] = (Psqrt[1]*Psqrt[1]') - (Psqrt[1]*Psqrt[1]')*(Nsqrt_0*Nsqrt_0')*(Psqrt[1]*Psqrt[1]')

    ## Save
    system.alpha = alpha
    system.V = V

    return system
end

# Fits a basic structural model to the response y, returning the smoothed
# estimates and their variances.
function state_space(y, s; X = Array{Float64}(0,0), nSeeds = 1)

    # create system struct
    system = create_system()
    system.X = X
    system.y = y
    system.s = s

    # check response dimensions
    system.n, system.p = size(y[:,:]) # number of observations and endogenous variables

    # check if there are explanatory variables
    n_obs_exp, n_exp = size(X[:,:])

    if n_obs_exp != system.n && n_obs_exp > 0
        println("Number of observations in X and y mismatch.")
    end

    println("Creating state space model with $(system.p) endogenous variable(s) and $n_exp exogenous variable(s).")

    system.m = system.p * (1 + system.s + n_exp)
    system.r = system.p * 3

    # measurement equation
    system.Z = Array{Array}(system.n)
    if n_exp > 0
        for t = 1 : system.n
            system.Z[t] = kron([X[t, :]' 1 0 1 zeros(1, system.s - 2)], eye(system.p, system.p))
        end
    else
        for t = 1 : system.n
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

    # Big Kappa initialization
    bigKappa = 1000

    ## Maximum likelihood estimation of parameters
    # number of unknown parameters
    nPsi = Int((1 + system.r/system.p)*(system.p*(system.p + 1)/2))
    # vector of unknowns
    bestPsi = Array{Float64}(nPsi)
    # search limits
    infLim = -1000.0
    supLim = 1000.0

    # optimization
    println("Starting maximum likelihood estimation.")

    if length(procs()) > 1 # parallel computing
        println("Running with $(length(procs())) procs in parallel.")
        # memory allocation
        seeds = SharedArray{Float64}(nPsi, nSeeds)
        bestPsi = SharedArray{Float64}(nPsi)
        logLikelihood = SharedArray{Float64}(nSeeds)
        psi = SharedArray{Float64}(nPsi, nSeeds)

        for iSeed = 1:nSeeds
            # uniformly generate values on the interval [infLim,supLim]
            seeds[:, iSeed] = infLim + rand(infLim:supLim, nPsi)
        end

        @sync @parallel for iSeed = 1:nSeeds
            # perform optimization
            println("Computing MLE for seed $iSeed of $nSeeds...")
            tryOpt = optimize(psiTilde -> state_space_likelihood(psiTilde, system), seeds[:, iSeed],
                                LBFGS(), Optim.Options(f_tol = 1e-6, g_tol = 1e-10, iterations = 10^5,
                                show_trace = true, show_every = 10^3))
            logLikelihood[iSeed] = tryOpt.minimum
            psi[:, iSeed] = tryOpt.minimizer
            println("Log-likelihood for seed $iSeed: $(-logLikelihood[iSeed])")
        end

        @show logLikelihood
        @show psi

    else # simple optimization
        println("Running with one proc.")
        # memory allocation
        seeds = Array{Float64}(nPsi, nSeeds)
        logLikelihood = Array{Float64}(nSeeds)
        psi = Array{Float64}(nPsi, nSeeds)

        for iSeed = 1:nSeeds
            # uniformly generate values on the interval [infLim,supLim]
            seeds[:, iSeed] = infLim + rand(infLim:supLim, nPsi)
            # perform optimization
            println("Computing MLE for seed $iSeed of $nSeeds...")
            tryOpt = optimize(psiTilde -> state_space_likelihood(psiTilde, system), seeds[:, iSeed],
                                LBFGS(), Optim.Options(f_tol = 1e-15, g_tol = 1e-15, x_tol = 1e-15,
                                iterations = 10^5, show_trace = true, show_every = 10^3))
            logLikelihood[iSeed] = tryOpt.minimum
            psi[:, iSeed] = tryOpt.minimizer
            println("Log-likelihood for seed $iSeed: $(-logLikelihood[iSeed])")
        end
    end

    println("Maximum likelihood estimation complete.")
    logLikelihood = -logLikelihood
    bestLogLik = maximum(logLikelihood)
    println("Log-Likelihood: $bestLogLik")
    # choosing the best feasible solution
    bestPsi = psi[:, indmax(logLikelihood)]
    system.psi = bestPsi

    # Measurement's covariance matrix
    if system.p > 1
        Hsqrt = tril(ones(system.p, system.p));
        nUnknownsH = Int(system.p * (system.p + 1) / 2)
        Hsqrt[Hsqrt .== 1] = bestPsi[1:nUnknownsH]
    else
        Hsqrt = bestPsi[1]
        nUnknownsH = 1
    end
    # State's covariance matrix
    Qsqrt = kron(eye(Int(system.r/system.p)), tril(ones(system.p, system.p)))
    Qsqrt[Qsqrt .== 1] = bestPsi[(nUnknownsH+1) : Int(nUnknownsH + (system.r/system.p)*(system.p*(system.p + 1)/2))]

    ## Filter & Smoother
    system = square_root_kalman_filter(Hsqrt, Qsqrt, system)
    system = square_root_smoother(system)

    system.Hsqrt = Hsqrt
    system.Qsqrt = Qsqrt

    ## Plots
    alphaMatrix = Array{Float64}(system.n, system.m)
    for t = 1 : system.n
        for i = 1 : system.m
            alphaMatrix[t,i] = system.alpha[t][i]
        end
    end
    y = system.y
    v = system.v
    system.level = Array{Array}(system.p)
    system.slope = Array{Array}(system.p)
    system.seasonal = Array{Array}(system.p)

    for i = 1 : system.p
        # Level
        system.level[i] = alphaMatrix[:,n_exp+i]

        # Slope
        system.slope[i] = alphaMatrix[:,n_exp+system.p+i]

        # Seasonal
        system.seasonal[i] = alphaMatrix[:,n_exp+2*system.p+i]
    end

    println("End of structural model estimation.")

    return system
end

function simulate_state_space(system, N, S; X = Array{Float64}(0,0))

    # check if there are explanatory variables
    n_obs_exp, n_exp = size(X[:,:])

    if N > n_obs_exp && n_obs_exp > 0
        println("Período de simulação é maior que período das explicativas")
    end

    Z = Array{Array}(N)
    if n_exp > 0
        for t = 1 : N
            Z[t] = kron([X[t, :]' 1 0 1 zeros(1, system.s - 2)], eye(system.p, system.p))
        end
    else
        for t = 1 : N
            Z[t] = kron([1 0 1 zeros(1, system.s - 2)], eye(system.p, system.p))
        end
    end

    # observation innovation
    dist_ϵ = MvNormal(zeros(system.p), sqrt.(system.Hsqrt*system.Hsqrt'))

    # state innovation
    dist_η = MvNormal(zeros(system.r), sqrt.(system.Qsqrt*system.Qsqrt'))

    α_sim = Array{Float64}(N,system.m)
    y_sim = Array{Array}(S)

    for s = 1 : S
        y_sim[s] = Array{Float64}(N, system.p)
        ϵ = rand(dist_ϵ, N)'
        η = rand(dist_η, N)'

        for t = 1 : N
            if t == 1
                α_sim[t,:] = system.T*system.alpha[system.n] + (system.R * η[t,:])
                y_sim[s][t,:] = Z[t]*α_sim[t,:] + ϵ[t,:]
            else
                α_sim[t,:] = system.T*α_sim[t-1,:] + (system.R * η[t,:])
                y_sim[s][t,:] = Z[t]*α_sim[t,:] + ϵ[t,:]
            end
        end
    end

    yMatrix = Array{Float64}(N, S)
    for t = 1 : N
        for s = 1 : S
            yMatrix[t,s] = y_sim[s][t]
        end
    end

    return yMatrix

end

# Saves plots for each state
function plots_state_space(system)

    n_obs_exp, n_exp = size(system.X[:,:])

    alphaMatrix = Array{Float64}(system.n, system.m)
    for t = 1 : system.n
        for i = 1 : system.m
            alphaMatrix[t,i] = system.alpha[t][i]
        end
    end

    for i = 1 : system.p
        # Time series
        p = plot(y[:,i], title = "Time series $i", label = "", ylims = [minimum(y[:,i])-1, maximum(y[:,i])+1])
        savefig(p, "series_$i.png")

        # Level
        p = plot(alphaMatrix[:,n_exp+i], title = "Level of series $i", label = "",
             ylims = [minimum(alphaMatrix[:,n_exp+i])-1, maximum(alphaMatrix[:,n_exp+i])+1])
        savefig(p, "level_$i.png")

        # Slope
        p = plot(alphaMatrix[:,n_exp+system.p+i], title = "Slope of series $i", label = "",
             ylims = [minimum(alphaMatrix[:,n_exp+system.p+i])-1, maximum(alphaMatrix[:,n_exp+system.p+i])+1])
        savefig(p, "slope_$i.png")

        # Seasonal
        p = plot(alphaMatrix[:,n_exp+2*system.p+i], title = "Seasonal of series $i", label = "",
             ylims = [minimum(alphaMatrix[:,n_exp+2*system.p+i])-1, maximum(alphaMatrix[:,n_exp+2*system.p+i])+1])
        savefig(p, "seasonal_$i.png")
    end

end
