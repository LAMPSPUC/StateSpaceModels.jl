export LBFGS, BFGS

function sample_seeds(model::StateSpaceModel{T}, n_seeds::Int) where T
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

     return seeds
end

function check_len_seeds(model::StateSpaceModel{T}, seeds::Vector{Vector{T}}) where T
    # Querying number of unknowns
    unknowns = Unknowns(model)
    n_psi = unknowns.n_unknowns

    for (i, seed) in enumerate(seeds)
        if length(seed) != n_psi
            error("Seed $i has $(length(seed)) elements and the model has $n_psi unknowns.")
        end
    end
    return 
end

mutable struct LBFGS{T <: Real} <: AbstractOptimizationMethod{T}
    f_tol::T
    g_tol::T
    iterations::Int
    n_seeds::Int
    seeds::Vector{Vector{T}}
    method::Optim.AbstractOptimizer
end


"""
    LBFGS(model::StateSpaceModel{T}, args...; kwargs...)

If an `Int` is provided the method will sample random seeds and use them as initial points for Optim LBFGS method.
If a `Vector{Vector{T}}` is provided it will use them as initial points for Optim LBFGS method.
"""
function LBFGS(model::StateSpaceModel{T}, n_seeds::Int; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                                        iterations::Int = 10^5) where T <: Real

    seeds = sample_seeds(model, n_seeds)
    
    return LBFGS{T}(f_tol, g_tol, iterations, n_seeds, seeds, Optim.LBFGS())
end

function LBFGS(model::StateSpaceModel{T}, seeds::Vector{Vector{T}}; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                                                    iterations::Int = 10^5) where T <: Real

    check_len_seeds(model, seeds)
    n_seeds = length(seeds)

    return LBFGS{T}(f_tol, g_tol, iterations, n_seeds, seeds, Optim.LBFGS())
end

mutable struct BFGS{T <: Real} <: AbstractOptimizationMethod{T}
    f_tol::T
    g_tol::T
    iterations::Int
    n_seeds::Int
    seeds::Vector{Vector{T}}
    method::Optim.AbstractOptimizer
end

"""
    BFGS(model::StateSpaceModel{T}, args...; kwargs...)

If an `Int` is provided the method will sample random seeds and use them as initial points for Optim BFGS method.
If a `Vector{Vector{T}}` is provided it will use them as initial points for Optim BFGS method.
"""
function BFGS(model::StateSpaceModel{T}, n_seeds::Int; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                                       iterations::Int = 10^5) where T <: Real

    seeds = sample_seeds(model, n_seeds)
 
    return BFGS{T}(f_tol, g_tol, iterations, n_seeds, seeds, Optim.BFGS())
end

function BFGS(model::StateSpaceModel{T}, seeds::Vector{Vector{T}}; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                                                    iterations::Int = 10^5) where T <: Real

    check_len_seeds(model, seeds)
    n_seeds = length(seeds)
    
    return BFGS{T}(f_tol, g_tol, iterations, n_seeds, seeds, Optim.BFGS())
end
