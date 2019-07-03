function compute_log_likelihood(n::Int, p::Int, v::Matrix{T}, F::Array{T, 3}, valid_insts::Vector{Int}) where T <: AbstractFloat
    log_likelihood::Float64 = n*p*log(2*pi)/2
    for t in valid_insts
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

function valid_instants(model::StateSpaceModel)
    valid_insts = Vector{Int}(undef, 0)
    for t = 1:model.dim.n
        if all(.!isnan.(model.y[t, :]))
            push!(valid_insts, t)
        end
    end
    return valid_insts
end

"""
    statespace_likelihood(psitilde::Vector{T}, model::StateSpaceModel) where T <: AbstractFloat

Compute log-likelihood concerning hyperparameter vector psitilde

Evaluate ``\\ell(\\psi)...`` TODO
"""
function statespace_likelihood(psitilde::Vector{T}, model::StateSpaceModel, filter_type::DataType) where T <: AbstractFloat
    # Calculate v and F 
    v, F = get_log_likelihood_params(psitilde, model, filter_type)
    # Compute log-likelihood based on v and F
    return compute_log_likelihood(model.dim.n, model.dim.p, v, F, valid_instants(model))
end

"""
TODO
"""
function estimate_statespace(model::StateSpaceModel, filter_type::DataType,
                             optimization_method::AbstractOptimizationMethod; verbose::Int = 1)
    error(optimization_method , " not implemented") # Returns an error if it cannot 
                                                    # find a specialized estimate_statespace
end