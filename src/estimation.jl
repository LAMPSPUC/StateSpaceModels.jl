function compute_log_likelihood(n::Int, p::Int, v::Matrix{T}, F::Array{T, 3}) where T <: AbstractFloat
    log_likelihood::Float64 = n*p*log(2*pi)/2
    for t = 1:n
        log_likelihood = log_likelihood + 0.5 * (logdet(F[:, :, t]) + (v[t, :]' * inv(F[:, :, t]) * v[t, :]))
    end
    return log_likelihood
end

function get_log_likelihood_params(psitilde::Vector{T}, model::StateSpaceModel,
                                   filter_type::DataType) where T <: AbstractFloat
    error(filter_type , " not implemented") # Returns an error if it cannot 
                                            # find a specialized get_log_likelihood_params
end

function statespace_covariance(psi::Vector{T}, p::Int, r::Int,
                               filter_type::DataType) where T <: AbstractFloat
    error(filter_type , " not implemented") # Returns an error if it cannot 
                                            # find a specialized statespace_covariance
end

"""
    statespace_likelihood(psitilde::Vector{T}, model::StateSpaceModel) where T <: AbstractFloat

Compute log-likelihood concerning hyperparameter vector psitilde

Evaluate ``\\ell(\\psi)...`` TODO
"""
function statespace_likelihood(psitilde::Vector{T}, model::StateSpaceModel) where T <: AbstractFloat
    # Calculate v and F 
    v, F = get_log_likelihood_params(psitilde, model, model.filter_type)
    # Compute log-likelihood based on v and F
    return compute_log_likelihood(model.dim.n, model.dim.p, v, F)
end

"""
    estimate_statespace(model::StateSpaceModel, nseeds::Int; f_tol::Float64 = 1e-10, g_tol::Float64 = 1e-10, iterations::Int = 10^5)

Estimate structural model hyperparameters
"""
function estimate_statespace(model::StateSpaceModel, nseeds::Int; f_tol::Float64 = 1e-10, g_tol::Float64 = 1e-10, 
                                iterations::Int = 10^5, verbose::Int = 1)

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

    if verbose > 0
        @info("Initiating maximum likelihood estimation with $(nseeds-1) seeds.")
    end

    # Generate initial values in [-1e3, 1e3]
    for iseed = 1:nseeds
        if iseed == 1
            seeds[:, iseed] = 1e-8*ones(npsi)
        else
            seeds[:, iseed] = rand(seedrange, npsi)
        end
    end

    # Optimization
    for iseed = 1:nseeds

        if verbose > 0
            @info("Optimizing likelihood for seed $(iseed-1) of $(nseeds-1)...")
            if iseed == 1
                @info("Seed 0 is aimed at degenerate cases.")
            end
        end

        optseed = optimize(psitilde -> statespace_likelihood(psitilde, model), seeds[:, iseed],
                            LBFGS(), Optim.Options(f_tol = f_tol, g_tol = g_tol, iterations = iterations,
                            show_trace = verbose == 2 ? true : false))
        loglikelihood[iseed] = -optseed.minimum
        psi[:, iseed] = optseed.minimizer

        if verbose > 0
            @info("Log-likelihood for seed $(iseed-1): $(loglikelihood[iseed])")
        end
    end

    if verbose > 0
        @info("Maximum likelihood estimation complete.")
        @info("Log-likelihood: $(maximum(loglikelihood))")
    end

    bestpsi = psi[:, argmax(loglikelihood)]
    H, Q    = statespace_covariance(bestpsi, model.dim.p, model.dim.r, model.filter_type)

    return StateSpaceCovariance(H, Q, model.filter_type)
end
