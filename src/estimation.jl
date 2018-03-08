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
