"""Computes sqrt-covariance matrices of the errors from hyperparameter vector psi"""
function statespace_covariance(psi::Vector{T}, p::Int, r::Int) where T <: AbstractFloat

    # Observation sqrt-covariance matrix
    if p > 1
        sqrtH     = tril!(ones(p, p))
        unknownsH = Int(p*(p + 1)/2)
        sqrtH[findall(x -> x == 1, sqrtH)] = psi[1:unknownsH]
    else
        sqrtH = psi[1].*ones(1, 1)
        unknownsH = 1
    end

    # State sqrt-covariance matrix
    sqrtQ = kron(Matrix{Float64}(I, Int(r/p), Int(r/p)), tril!(ones(p, p)))
    sqrtQ[findall(x -> x == 1, sqrtQ)] = psi[(unknownsH+1):Int(unknownsH + (r/p)*(p*(p + 1)/2))]

    return sqrtH, sqrtQ
end

"""Computes log-likelihood concerning hyperparameter vector psitilde"""
function statespace_likelihood(psitilde::Vector{Float64}, sys::StateSpaceSystem)

    sqrtH, sqrtQ = statespace_covariance(psitilde, sys.dim.p, sys.dim.r)

    # Obtain innovation v and its variance F
    ss_filter = sqrt_kalmanfilter(sys, sqrtH, sqrtQ)

    # Check if steady state was attained
    if ss_filter.tsteady < sys.dim.n
        for t = ss_filter.tsteady+1:sys.dim.n
            ss_filter.sqrtF[t] = ss_filter.sqrtF[ss_filter.tsteady]
        end
    end

    # Compute log-likelihood based on v and F
    loglikelihood = sys.dim.n*sys.dim.p*log(2*pi)/2
    for t = sys.dim.m:sys.dim.n
        det_sqrtF = det(ss_filter.sqrtF[t]*ss_filter.sqrtF[t]')
        if det_sqrtF < 1e-30
            det_sqrtF = 1e-30
        end
        loglikelihood = loglikelihood + .5 * (log(det_sqrtF) +
                        (ss_filter.v[t]' * pinv(ss_filter.sqrtF[t]*ss_filter.sqrtF[t]') * ss_filter.v[t])[1])
    end

    return loglikelihood
end

"""Estimates structural model hyperparameters"""
function estimate_statespace(sys::StateSpaceSystem, nseeds::Int; f_tol = 1e-10, g_tol = 1e-10, iterations = 10^5)

    nseeds += 1 # creating additional seed for degenerate cases

    # Initialization
    npsi          = Int((1 + sys.dim.r/sys.dim.p)*(sys.dim.p*(sys.dim.p + 1)/2))
    seeds         = Matrix{Float64}(undef, npsi, nseeds)
    loglikelihood = Vector{Float64}(undef, nseeds)
    psi           = Matrix{Float64}(undef, npsi, nseeds)

    # Initial conditions
    inflim    = -1e3
    suplim    = 1e3
    seedrange = collect(inflim:0.1:suplim)

    # Avoiding zero values for covariance
    deleteat!(seedrange, findall(x -> x == 0.0, seedrange))

    @info("Initiating maximum likelihood estimation with $(nseeds-1) seeds.")

    # Generate initial values in [inflim, suplim]
    for iseed = 1:nseeds
        if iseed == 1
            seeds[:, iseed] = 1e-8*ones(npsi)
        else
            seeds[:, iseed] = rand(seedrange, npsi)
        end
    end

    # Optimization
    for iseed = 1:nseeds
        @info("Optimizing likelihood for seed $(iseed-1) of $(nseeds-1)...")
        if iseed == 1
            @info("Seed 0 is aimed at degenerate cases.")
        end
        optseed = optimize(psitilde -> statespace_likelihood(psitilde, sys), seeds[:, iseed],
                            LBFGS(), Optim.Options(f_tol = f_tol, g_tol = g_tol, iterations = iterations,
                            show_trace = false))
        loglikelihood[iseed] = -optseed.minimum
        psi[:, iseed] = optseed.minimizer
        @info("Log-likelihood for seed $(iseed-1): $(loglikelihood[iseed])")
    end

    @info("Maximum likelihood estimation complete.")
    @info("Log-likelihood: $(maximum(loglikelihood))")

    bestpsi      = psi[:, argmax(loglikelihood)]
    sqrtH, sqrtQ = statespace_covariance(bestpsi, sys.dim.p, sys.dim.r)

    # Parameter structure
    param = StateSpaceParameters(sqrtH, sqrtQ)

    return param
end
