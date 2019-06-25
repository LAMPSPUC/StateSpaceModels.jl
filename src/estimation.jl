function compute_log_likelihood(n::Int, p::Int, m::Int, v::Matrix{T}, F::Array{T, 3}) where T <: AbstractFloat
    log_likelihood = n*p*log(2*pi)/2
    for t = m+1:n
        @info("F: ", F[:, :, t])
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
    return compute_log_likelihood(model.dim.n, model.dim.p, model.dim.m, v, F)
end

"""
TODO
"""
function estimate_statespace(model::StateSpaceModel, 
                             optimization_method::AbstractOptimizationMethod; verbose::Int = 1)
    error(optimization_method , " not implemented") # Returns an error if it cannot 
                                                    # find a specialized estimate_statespace
end