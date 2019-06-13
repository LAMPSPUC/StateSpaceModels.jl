"""
    statespace_covariance(psi::Vector{T}, p::Int, r::Int) where T <: AbstractFloat

Compute sqrt-covariance matrices of the errors from hyperparameter vector psi
"""
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

"""
    statespace_likelihood(psitilde::Vector{T}, model::StateSpaceModel) where T <: AbstractFloat

Compute log-likelihood concerning hyperparameter vector psitilde
"""
function statespace_likelihood(psitilde::Vector{T}, model::StateSpaceModel) where T <: AbstractFloat

    sqrtH, sqrtQ = statespace_covariance(psitilde, model.dim.p, model.dim.r)

    # Obtain innovation v and its variance F
    kfilter, U2star, K = sqrt_kalmanfilter(model, sqrtH, sqrtQ)

    # Compute log-likelihood based on v and F
    F =  

    return evaluate_likelihood(model.dim.n, model.dim.p, kfilter.v, F)
end

"""
"""
function evaluate_likelihood(n::Int, p::Int, v::Matrix{T}, F::Array{T, 3}) where T <: AbstractFloat
    loglikelihood::Float64 = n*p*log(2*pi)/2
    for t = 1:n
        loglikelihood = loglikelihood + 0.5 * (logdet(F) + (v[t, :]' * inv(F[:, :, t]) * v[t, :]))
    end
    return loglikelihood
end

"""

"""
function evaluate_F(sqrtF::Array{T, 3}, n::Int) where T <: AbstractFloat
    F = similar(sqrtF)
    for t = 1:n

        F[:, :, t] = LinearAlgebra.BLAS.trmm('R', 'L', 'T', 'N', 1.0, sqrtF[:, :, t], sqrtF[:, :, t])
    end
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
    deleteat!(seedrange, findall(x -> x == 0.0, seedrange))

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

    bestpsi      = psi[:, argmax(loglikelihood)]
    sqrtH, sqrtQ = statespace_covariance(bestpsi, model.dim.p, model.dim.r)

    # Parameter structure
    param = StateSpaceParameters(sqrtH, sqrtQ)

    return param
end
