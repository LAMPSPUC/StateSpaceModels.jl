function compute_log_likelihood(n::Int, p::Int, v::Matrix{T}, F::Array{T, 3}, valid_insts::Vector{Int}) where T <: AbstractFloat
    log_likelihood::Float64 = n*p*log(2*pi)/2
    @inbounds @views for t in valid_insts
        log_likelihood += 0.5 * (logdet(F[:, :, t]) + (v[t, :]' * invertF(F[:, :, t]) * v[t, :]))
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
    statespace_likelihood(psitilde::Vector{T}, model::StateSpaceModel) where T <: AbstractFloat

Compute log-likelihood concerning hyperparameter vector psitilde (``\\psi``)

Evaluate ``\\ell(\\psi;y_n)= -\\frac{np}{2}\\log2\\pi - \\frac{1}{2} \\sum_{t=1}^n \\log |F_t| - 
\\frac{1}{2} \\sum_{t=1}^n v_t^{T} F_t^{-1} v_t ``
"""
function statespace_likelihood(psitilde::Vector{T}, model::StateSpaceModel, 
                               valid_insts::Vector{Int}, filter_type::DataType) where T <: AbstractFloat
    # Calculate v and F 
    v, F = get_log_likelihood_params(psitilde, model, filter_type)
    # Compute log-likelihood based on v and F
    return compute_log_likelihood(model.dim.n, model.dim.p, v, F, valid_insts)
end

"""
    estimate_statespace(model::StateSpaceModel, filter_type::DataType,
                        optimization_method::AbstractOptimizationMethod; verbose::Int = 1)

Estimate parameters of the `StateSpaceModel` according to its `filter_type` and `optimization_method`.
"""
function estimate_statespace(model::StateSpaceModel, filter_type::DataType,
                             optimization_method::AbstractOptimizationMethod; verbose::Int = 1)
    error(optimization_method , " not implemented") # Returns an error if it cannot 
                                                    # find a specialized estimate_statespace
end