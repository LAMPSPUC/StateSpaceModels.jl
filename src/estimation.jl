"""Computes sqrt-covariance matrices of the errors from hyperparameter vector psi"""
function statespace_covariance(psi::Array{Float64,1}, p::Int, r::Int)

    # Observation sqrt-covariance matrix
    if p > 1
        sqrtH = tril(ones(p, p))
        unknownsH = Int(p*(p + 1)/2)
        sqrtH[sqrtH .== 1] = psi[1:unknownsH]
    else
        sqrtH = [psi[1]]
        unknownsH = 1
    end

    # State sqrt-covariance matrix
    sqrtQ = kron(eye(Int(r/p)), tril(ones(p, p)))
    sqrtQ[sqrtQ .== 1] = psi[(unknownsH+1):Int(unknownsH + (r/p)*(p*(p + 1)/2))]

    return sqrtH, sqrtQ
end

"""Computes log-likelihood concerning hyperparameter vector psitilde"""
function statespace_likelihood(psitilde::Array{Float64,1}, sys::StateSpaceSystem, dim::StateSpaceDimensions)

    sqrtH, sqrtQ = statespace_covariance(psitilde, dim.p, dim.r)

    # Obtain innovation v and its variance F
    ss_filter = sqrt_kalmanfilter(sys, dim, sqrtH, sqrtQ)

    # Check if steady state was attained
    if ss_filter.tsteady < dim.n
        for t = ss_filter.tsteady+1:dim.n
            ss_filter.sqrtF[t] = ss_filter.sqrtF[ss_filter.tsteady]
        end
    end

    # Compute log-likelihood based on v and F
    loglikelihood = dim.n*dim.p*log(2*pi)/2
    for t = dim.m:dim.n
        det_sqrtF = det(ss_filter.sqrtF[t]*ss_filter.sqrtF[t]')
        if det_sqrtF < 1e-30
            det_sqrtF = 1e-30
        end
        loglikelihood = loglikelihood + .5 * (log(det_sqrtF) +
                         ss_filter.v[t]' * pinv(ss_filter.sqrtF[t]*ss_filter.sqrtF[t]') * ss_filter.v[t])
    end

    return loglikelihood
end

"""Estimates structural model hyperparameters"""
function estimate_statespace(sys::StateSpaceSystem, dim::StateSpaceDimensions, nseeds::Int;
    f_tol = 1e-10, g_tol = 1e-10, iterations = 10^5)

    # Initialization
    npsi = Int((1 + dim.r/dim.p)*(dim.p*(dim.p + 1)/2))
    seeds = SharedArray{Float64}(npsi, nseeds)
    loglikelihood = SharedArray{Float64}(nseeds)
    psi = SharedArray{Float64}(npsi, nseeds)

    # Initial conditions
    inflim = -1e3
    suplim = 1e3
    seedrange = collect(inflim:0.1:suplim)

    # Avoiding zero values for covariance
    deleteat!(seedrange, find(seedrange .== 0.0))

    info("Initiating maximum likelihood estimation with $nseeds seeds.")
    info("Running with $(length(procs())) threads.")

    # Generate initial values in [inflim, suplim]
    for iseed = 1:nseeds
        if iseed == 1
            seeds[:, iseed] = 1e-8*ones(npsi)
        else
            seeds[:, iseed] = rand(seedrange, npsi)
        end
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
