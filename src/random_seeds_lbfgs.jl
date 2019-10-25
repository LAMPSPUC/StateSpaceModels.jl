"""
TODO
"""
mutable struct RandomSeedsLBFGS{T <: Real} <: AbstractOptimizationMethod
    f_tol::T
    g_tol::T
    iterations::Int
    nseeds::Int
    seeds::Array{T}

    function RandomSeedsLBFGS(; f_tol::T = 1e-6, g_tol::T = 1e-6, iterations::Int = 10^5, nseeds::Int = 3) where T <: Real
        return new{T}(f_tol, g_tol, 1e5, nseeds)
    end
end

function estimate_statespace!(model::StateSpaceModel{T}, filter_type::DataType,
                             opt_method::RandomSeedsLBFGS; verbose::Int = 1) where T

    nseeds = opt_method.nseeds                         
    nseeds += 1 # creating additional seed for degenerate cases

    # Initialization
    unknowns = Unknowns(model)

    if unknowns.n_unknowns == 0
        @warn("StateSpaceModel has no parameters to be estimated.")
        return model
    end
 
    npsi          = unknowns.n_unknowns
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
        model_opt = copy(model)
        try
            optseed = optimize(psitilde -> statespace_likelihood(psitilde, model_opt, unknowns, valid_insts, filter_type), seeds[:, iseed],
                                LBFGS(), Optim.Options(f_tol = opt_method.f_tol, 
                                                       g_tol = opt_method.g_tol, 
                                                       iterations = opt_method.iterations,
                                                       show_trace = (verbose == 3 ? true : false) ))
            loglikelihood[iseed] = -optseed.minimum
            psi[:, iseed] = optseed.minimizer
            push!(optseeds, optseed)

            print_loglikelihood(verbose, iseed, loglikelihood, t0)
        catch
            println("seed diverged")
        end
    end

    best_seed = argmax(loglikelihood)

    print_estimation_end(verbose, loglikelihood)
    verbose >= 2 && println("Best seed optimization result: \n $(optseeds[best_seed])")

    # Put seeds in the model struct
    opt_method.seeds = seeds
    # Take the best seed and fill the model with it
    bestpsi = psi[:, best_seed]
    fill_model_with_parameters!(model, bestpsi, unknowns)

    return 
end
