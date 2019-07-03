function estimate_statespace(model::StateSpaceModel, 
                             opt_method::RandomSeedsLBFGS; verbose::Int = 1)

    nseeds = opt_method.nseeds                         
    nseeds += 1 # creating additional seed for degenerate cases

    # Initialization
    npsi          = Int((1 + model.dim.r/model.dim.p)*(model.dim.p*(model.dim.p + 1)/2))
    seeds         = Matrix{Float64}(undef, npsi, nseeds)
    loglikelihood = Vector{Float64}(undef, nseeds)
    psi           = Matrix{Float64}(undef, npsi, nseeds)

    # Initial conditions
    seedrange = collect(-1e3:0.1:1e3)

    # Avoiding zero values for covariance
    deleteat!(seedrange, findall(iszero, seedrange))

    print_estimation_start(verbose, nseeds)

    # Generate initial values in [-1e3, 1e3]
    for iseed = 1:nseeds
        if iseed == 1
            seeds[:, iseed] = 1e-8*ones(npsi)
        else
            seeds[:, iseed] = rand(seedrange, npsi)
        end
    end

    t0 = now()

    # Optimization
    for iseed = 1:nseeds
        optseed = optimize(psitilde -> statespace_likelihood(psitilde, model), seeds[:, iseed],
                            LBFGS(), Optim.Options(f_tol = opt_method.f_tol, 
                                                   g_tol = opt_method.g_tol, 
                                                   iterations = opt_method.iterations,
                                                   show_trace = (verbose == 2 ? true : false) ))
        loglikelihood[iseed] = -optseed.minimum
        psi[:, iseed] = optseed.minimizer

        print_loglikelihood(verbose, iseed, loglikelihood, t0)
    end

    print_estimation_end(verbose, loglikelihood)

    # Put seeds in the model struct
    model.optimization_method.seeds = seeds

    bestpsi = psi[:, argmax(loglikelihood)]
    H, Q    = statespace_covariance(bestpsi, model.dim.p, model.dim.r, model.filter_type)

    return StateSpaceCovariance(H, Q, model.filter_type)
end
