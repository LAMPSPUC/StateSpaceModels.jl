export BFGS

mutable struct BFGS{T <: Real} <: AbstractOptimizationMethod
    f_tol::T
    g_tol::T
    iterations::Int
    n_seeds::Int
    seeds::Vector{Vector{T}}
end

"""
    BFGS(model::StateSpaceModel{T}, args...; kwargs...)

If an `Int` is provided the method will sample random seeds and use them as initial points for Optim BFGS method.
If a `Vector{Vector{T}}` is provided it will use them as initial points for Optim BFGS method.
"""
function BFGS(model::StateSpaceModel{T}, n_seeds::Int; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                                       iterations::Int = 10^5) where T <: Real

    # Querying number of unknowns
    unknowns = Unknowns(model)
    n_psi = unknowns.n_unknowns

    seeds = Vector{Vector{T}}(undef, n_seeds)

    seedrange = collect(T, -1e3:0.1:1e3)

    # Avoiding zero values for covariance
    deleteat!(seedrange, findall(iszero, seedrange))

    # Generate initial values in [-1e3, 1e3]
    for i = 1:n_seeds
        seeds[i] = rand(seedrange, n_psi)
    end
    
    return BFGS{T}(f_tol, g_tol, iterations, n_seeds, seeds)
end

function BFGS(model::StateSpaceModel{T}, seeds::Vector{Vector{T}}; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                                                    iterations::Int = 10^5) where T <: Real

    # Querying number of unknowns
    unknowns = Unknowns(model)
    n_psi = unknowns.n_unknowns

    n_seeds = length(seeds)

    for (i, seed) in enumerate(seeds)
        if length(seed) != n_psi
            error("Seed $i has $(length(seed)) elements and the model has $n_psi unknowns.")
        end
    end
    
    return BFGS{T}(f_tol, g_tol, iterations, n_seeds, seeds)
end

function estimate_statespace!(model::StateSpaceModel{T}, filter_type::DataType,
                              opt_method::BFGS{T}; verbose::Int = 1) where T


    unknowns = Unknowns(model)

    if unknowns.n_unknowns == 0
        @warn("StateSpaceModel has no parameters to be estimated.")
        return model
    end
 
    n_psi          = unknowns.n_unknowns
    loglikelihood = Vector{T}(undef, opt_method.n_seeds)
    psi           = Matrix{T}(undef, n_psi, opt_method.n_seeds)
    optseeds      = Vector{Optim.OptimizationResults}(undef, 0)

    print_estimation_start(verbose, opt_method.n_seeds)

    # compute the valid instants to evaluate loglikelihood
    valid_insts = valid_instants(model)

    t0 = now()

    # Optimization
    for (i, seed) in enumerate(opt_method.seeds)
        try
            optseed = optimize(psitilde -> statespace_likelihood(psitilde, model, unknowns, valid_insts, filter_type), seed,
                                Optim.BFGS(), Optim.Options(f_tol = opt_method.f_tol, 
                                                            g_tol = opt_method.g_tol, 
                                                            iterations = opt_method.iterations,
                                                            show_trace = (verbose == 3 ? true : false) ))
            loglikelihood[i] = -optseed.minimum
            psi[:, i] = optseed.minimizer
            push!(optseeds, optseed)

            print_loglikelihood(verbose, i, loglikelihood, t0)
        catch err
            println(err)
            println("seed diverged")
        end
    end

    best_seed = argmax(loglikelihood)

    print_estimation_end(verbose, loglikelihood)
    verbose >= 2 && println("Best seed optimization result: \n $(optseeds[best_seed])")

    # Take the best seed and fill the model with it
    bestpsi = psi[:, best_seed]
    fill_model_with_parameters!(model, bestpsi, unknowns)

    return 
end
