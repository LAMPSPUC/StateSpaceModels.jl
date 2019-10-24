function estimate_statespace(model::StateSpaceModel{T}, filter_type::DataType,
                             opt_method::RandomSeedsLBFGS; verbose::Int = 1) where T

    nseeds = opt_method.nseeds                         
    nseeds += 1 # creating additional seed for degenerate cases

    # Initialization
    npsi          = Int((1 + model.dim.r/model.dim.p)*(model.dim.p*(model.dim.p + 1)/2))
    seeds         = Matrix{T}(undef, npsi, nseeds)
    loglikelihood = Vector{T}(undef, nseeds)
    psi           = Matrix{T}(undef, npsi, nseeds)
    optseeds      = Vector{Optim.OptimizationResults}(undef, 0)

    # Initial conditions
    seedrange = collect(T, -1e3:0.1:1e3)

    # Avoiding zero values for covariance
    deleteat!(seedrange, findall(iszero, seedrange))

    print_estimation_start(verbose, nseeds)

    # Generate initial values in [-1e3, 1e3]
    for iseed = 1:nseeds
        if iseed == 1
            seeds[:, iseed] = T(1e-8)*ones(T, npsi)
        else
            seeds[:, iseed] = rand(seedrange, npsi)
        end
    end

    # compute the valid instants to evaluate loglikelihood
    valid_insts = valid_instants(model)

    t0 = now()

    # Optimization
    for iseed = 1:nseeds
        optseed = optimize(psitilde -> statespace_likelihood(psitilde, model, valid_insts, filter_type), seeds[:, iseed],
                            LBFGS(), Optim.Options(f_tol = opt_method.f_tol, 
                                                   g_tol = opt_method.g_tol, 
                                                   iterations = opt_method.iterations,
                                                   show_trace = (verbose == 3 ? true : false) ))
        loglikelihood[iseed] = -optseed.minimum
        psi[:, iseed] = optseed.minimizer
        push!(optseeds, optseed)

        print_loglikelihood(verbose, iseed, loglikelihood, t0)
    end

    best_seed = argmax(loglikelihood)

    print_estimation_end(verbose, loglikelihood)
    verbose >= 2 && println("Best seed optimization result: \n $(optseeds[best_seed])")

    # Put seeds in the model struct
    opt_method.seeds = seeds

    bestpsi = psi[:, best_seed]
    H, Q    = statespace_covariance(bestpsi, model.dim.p, model.dim.r, filter_type)

    return StateSpaceCovariance(H, Q, filter_type)
end
