export size, ztr

"""
    size(model::StateSpaceModel)

Return the dimensions `n`, `p`, `m` and `r` of the `StateSpaceModel`
"""
function size(model::StateSpaceModel)
    return model.dim.n, model.dim.p, model.dim.m, model.dim.r
end

"""
    ztr(model::StateSpaceModel)

Return the state space model arrays `Z`, `T` and `R` of the `StateSpaceModel`
"""
function ztr(model::StateSpaceModel)
    return model.Z, model.T, model.R
end


# Linar Algebra wrappers
function gram_in_time(mat::Array{Float64, 3})
    gram_in_time = similar(mat)
    @inbounds @views for t = 1:size(gram_in_time, 3)
        gram_in_time[:, :, t] = gram(mat[:, :, t])
    end
    return gram_in_time
end

function gram(mat::AbstractArray{T}) where T <: AbstractFloat
    return LinearAlgebra.BLAS.gemm('N', 'T', mat, mat) # mat*mat'    
end

"""
    check_steady_state(P_t1::Matrix{T}, P_t::Matrix{T}, tol::T) where T <: AbstractFloat

Return `true` if steady state was attained with respect to tolerance `tol`, `false` otherwise
"""
function check_steady_state(P_t1::Matrix{T}, P_t::Matrix{T}, tol::T) where T <: AbstractFloat
    return maximum(abs.((P_t1 - P_t)./P_t1)) < tol ? true : false
end

function check_steady_state(P::AbstractArray{T}, t::Int, tol::T) where T <: AbstractFloat
    @inbounds for j in axes(P, 2), i in axes(P, 1)
        if abs((P[i, j, t+1] - P[i, j, t])/P[i, j, t+1]) > tol
            return false
        end
    end
    return true
end

"""
    ensure_pos_sym!(M::Matrix{T}; ϵ::T = 1e-8) where T <: AbstractFloat

Ensure that matrix `M` is positive and symmetric to avoid numerical errors when numbers are small by doing `(M + M')/2 + ϵ*I`
"""
function ensure_pos_sym!(M::AbstractArray{T}, t::Int; ϵ::T = 1e-8) where T <: AbstractFloat
    @inbounds for j in axes(M, 2), i in 1:j
        if i == j
            M[i, i, t] = (M[i, i, t] + M[i, i, t])/2 + ϵ
        else
            M[i, j, t] = (M[i, j, t] + M[j, i, t])/2
            M[j, i, t] = M[i, j, t]
        end
    end
    return 
end

function sum_matrix!(mat_prin::AbstractArray{T}, mat_sum::AbstractMatrix{T}, t::Int, offset::Int) where T <:AbstractFloat
    @inbounds for j in axes(mat_prin, 2), i in axes(mat_prin, 1)
        mat_prin[i, j, t + offset] = mat_prin[i, j, t + offset] + mat_sum[i, j]
    end
    return 
end

function sum_matrix!(mat_prin::AbstractArray{T}, mat_sum::AbstractArray{T}, t::Int, offset::Int) where T <:AbstractFloat
    @inbounds for j in axes(mat_prin, 2), i in axes(mat_prin, 1)
        mat_prin[i, j, t + offset] = mat_prin[i, j, t + offset] + mat_sum[i, j, t]
    end
    return 
end

function invertF(F::AbstractMatrix{T}) where T
    return size(F, 1) == 1 ? inv.(F) : inv(F)
end

function check_missing_observation(y::Matrix{T}) where T
    missing_obs = Vector{Int}(undef, 0)
    for i in axes(y, 1), j in axes(y, 2)
        if isnan(y[i, j])
            # If is empty or if the index i is not already pushed
            (isempty(missing_obs) || (missing_obs[end] != i)) && push!(missing_obs, i)
        end
    end
    return missing_obs
end

function Base.show(io::IO, ss::StateSpace)
    println("An estimated state-space model with")
    println("filter_type = $(ss.filter_type),")
    println("optimization_method = $(typeof(ss.optimization_method)).")
    return nothing
end

function Base.show(io::IO, model::StateSpaceModel)
    println("A $(model.mode) state-space model with")
    println("n = $(model.dim.n),")
    println("p = $(model.dim.p),")
    println("m = $(model.dim.m),")
    println("r = $(model.dim.r).")
    return nothing
end