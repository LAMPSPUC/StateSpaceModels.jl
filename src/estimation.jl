function compute_log_likelihood(n::Int, p::Int, v::Matrix{T}, F::Array{T, 3}, valid_insts::Vector{Int}) where T
    if p == 1
        return compute_log_likelihood_univariate(n, v, F, valid_insts)
    else
        return compute_log_likelihood_multivariate(n, p, v, F, valid_insts)
    end
end

function compute_log_likelihood(n::Int, p::Int, v::Vector{T}, F::Vector{T}, valid_insts::Vector{Int}) where T
    log_likelihood::T = n*log(2*pi)/2
    @inbounds for t in valid_insts
        log_likelihood += 0.5 * (log(F[t]) + (v[t]^2)/F[t])
    end
    return log_likelihood
end

function compute_log_likelihood_univariate(n::Int, v::Matrix{T}, F::Array{T, 3}, valid_insts::Vector{Int}) where T
    log_likelihood::T = n*log(2*pi)/2
    @inbounds for t in valid_insts
        log_likelihood += 0.5 * (log(F[1, 1, t]) + (v[t, 1]^2)/F[1, 1, t])
    end
    return log_likelihood
end

function compute_log_likelihood_multivariate(n::Int, p::Int, v::Matrix{T}, F::Array{T, 3}, valid_insts::Vector{Int}) where T 
    log_likelihood::T = n*p*log(2*pi)/2
    @inbounds @views for t in valid_insts
        log_likelihood += 0.5 * (logdetF(F, t) + (v[t, :]' * invertF(F, t) * v[t, :]))
    end
    return log_likelihood
end

function get_log_likelihood_params(model::StateSpaceModel{T},
                                   filter_type::DataType) where T
    error(filter_type , " not implemented") # Returns an error if it cannot 
                                            # find a specialized get_log_likelihood_params
end

function valid_instants(model::StateSpaceModel{T}) where T
    valid_insts = collect(1:model.dim.n)
    offset::Int = 0 # Increase as we delete
    for t in axes(model.y, 1), j in axes(model.y, 2)
        if isnan(model.y[t, j])
            deleteat!(valid_insts, t - offset)
            offset += 1
        end
    end
    return valid_insts
end

"""
    statespace_loglik(psitilde::Vector{T}, model::StateSpaceModel, unknowns::Unknowns, 
                            valid_insts::Vector{Int}, filter_type::DataType) where T

Compute log-likelihood concerning hyperparameter vector psitilde (``\\psi``)

Evaluate ``\\ell(\\psi;y_n)= -\\frac{np}{2}\\log2\\pi - \\frac{1}{2} \\sum_{t=1}^n \\log |F_t| - 
\\frac{1}{2} \\sum_{t=1}^n v_t^{\\top} F_t^{-1} v_t ``
"""
function statespace_loglik(psitilde::Vector{T}, model::StateSpaceModel, unknowns::Unknowns, 
                               valid_insts::Vector{Int}, filter_type::DataType) where T
    # Fill all psitilde in the model
    fill_model_with_parameters!(model, psitilde, unknowns)
    # Calculate v and F 
    v, F = get_log_likelihood_params(model, filter_type)
    # Compute log-likelihood based on v and F
    return compute_log_likelihood(model.dim.n, model.dim.p, v, F, valid_insts)
end

"""
    estimate_statespace(model::StateSpaceModel{T}, filter_type::DataType,
                            optimization_method::AbstractOptimizationMethod; verbose::Int = 1) where T

Estimate parameters of the `StateSpaceModel` according to its `filter_type` and `optimization_method`.
"""
function estimate_statespace(model::StateSpaceModel{T}, filter_type::DataType,
                             optimization_method::AbstractOptimizationMethod; verbose::Int = 1) where T
    error(optimization_method , " not implemented") # Returns an error if it cannot 
                                                    # find a specialized estimate_statespace
end

function AIC(n_unknowns::Int, log_lik::T) where T
    return T(2 * n_unknowns - 2 * log_lik)
end

function BIC(n::Int, n_unknowns::Int, log_lik::T) where T
    return T(log(n) * n_unknowns - 2 * log_lik)
end

function estimate_statespace!(model::StateSpaceModel{T}, filter_type::DataType,
                              opt_method::AbstractOptimizationMethod{T}; verbose::Int = 1) where T


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
            optseed = optimize(psitilde -> statespace_loglik(psitilde, model, unknowns, valid_insts, filter_type), seed,
                            opt_method.method, Optim.Options(f_tol = opt_method.f_tol, 
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

    log_lik, best_seed = findmax(loglikelihood)

    aic = AIC(n_psi, log_lik)
    bic = BIC(length(model.y), n_psi, log_lik)

    print_estimation_end(verbose, log_lik, aic, bic)
    verbose >= 2 && println("Best seed optimization result: \n $(optseeds[best_seed])")

    # Take the best seed and fill the model with it
    bestpsi = psi[:, best_seed]
    fill_model_with_parameters!(model, bestpsi, unknowns)

    return 
end