export size, ztr, copy, statespace_recursion

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

"""
    statespace_recursion(model::StateSpaceModel{Typ}, N::Int, initial_a::Matrix{Typ}) where Typ

Runs the recursion of a state space model ignoring the y.
"""
function statespace_recursion(model::StateSpaceModel{Typ}, initial_a::Matrix{Typ}) where Typ

    # If model has any unknows is not possible to perform the recursion
    unknowns = Unknowns(model)
    if unknowns.n_unknowns != 0
        error("StateSpaceModel has unknown parameters.")
    end

    # Save the distributions H and Q
    dist_H = MvNormal(model.H)
    dist_Q = MvNormal(model.Q)

    p, m, n = size(model.Z)

    if size(initial_a, 2) != m
        error("intial_a must be a 1 by $m matrix.")
    end

    y = Matrix{Typ}(undef, n, p)
    α = Matrix{Typ}(undef, n, m)
    α[1, :] = initial_a

    for t = 1:n-1
        y[t, :]   = model.Z[:, :, t]*α[t, :] + model.d[t, :] + rand(dist_H)
        α[t+1, :] = model.T*α[t, :] + model.c[t, :] + model.R*rand(dist_Q)
    end
    y[n, :] = model.Z[:, :, n]*α[n, :] +  model.d[n, :] + rand(dist_H)

    return y, α
end


# Linar Algebra wrappers
function gram_in_time(mat::Array{T, 3}) where T
    gram_in_time = similar(mat)
    @inbounds @views for t = 1:size(gram_in_time, 3)
        gram_in_time[:, :, t] = gram(mat[:, :, t])
    end
    return gram_in_time
end

function gram(mat::AbstractArray{T}) where T
    if size(mat, 1) == 1
        gram_mat = Matrix{T}(undef, 1, 1)
        gram_mat[1, 1] = mat[1, 1]^2
        return gram_mat
    else
        return LinearAlgebra.BLAS.gemm('N', 'T', mat, mat) # mat*mat'
    end
end

"""
    check_steady_state(P_t1::Matrix{T}, P_t::Matrix{T}, tol::T) where T

Return `true` if steady state was attained with respect to tolerance `tol`, `false` otherwise.
The steady state is checked by the following equation `maximum(abs.((P_t1 - P_t)./P_t1)) < tol`.
"""
function check_steady_state(P_t1::Matrix{T}, P_t::Matrix{T}, tol::T) where T
    return maximum(abs.((P_t1 - P_t)./P_t1)) < tol ? true : false
end

function check_steady_state(P::AbstractArray{T}, t::Int, tol::T) where T
    @inbounds for j in axes(P, 2), i in axes(P, 1)
        if abs((P[i, j, t+1] - P[i, j, t])/P[i, j, t+1]) > tol
            return false
        end
    end
    return true
end

"""
    ensure_pos_sym!(M::Matrix{T}; ϵ::T = 1e-8) where T

Ensure that matrix `M` is positive and symmetric to avoid numerical errors when numbers are small by doing `(M + M')/2 + ϵ*I`
"""
function ensure_pos_sym!(M::AbstractArray{T}, t::Int; ϵ::T = T(1e-8)) where T
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

function ensure_pos_sym!(M::AbstractArray{T}; ϵ::T = T(1e-8)) where T
    @inbounds for j in axes(M, 2), i in 1:j
        if i == j
            M[i, i] = (M[i, i] + M[i, i])/2 + ϵ
        else
            M[i, j] = (M[i, j] + M[j, i])/2
            M[j, i] = M[i, j]
        end
    end
    return
end

function sum_matrix!(mat_prin::AbstractArray{T}, mat_sum::AbstractMatrix{T}, t::Int, offset::Int) where T
    @inbounds for j in axes(mat_prin, 2), i in axes(mat_prin, 1)
        mat_prin[i, j, t + offset] = mat_prin[i, j, t + offset] + mat_sum[i, j]
    end
    return
end

function sum_matrix!(mat_prin::AbstractArray{T}, mat_sum::AbstractArray{T}, t::Int, offset::Int) where T
    @inbounds for j in axes(mat_prin, 2), i in axes(mat_prin, 1)
        mat_prin[i, j, t + offset] = mat_prin[i, j, t + offset] + mat_sum[i, j, t]
    end
    return
end

function invertF(F::Array{T, 3}, t::Int) where T
    return @inbounds @views size(F, 1) == 1 ? 1/(F[1, 1, t]) : inv(F[:, :, t])
end

function invertF(F::Vector{T}, t::Int) where T
    return 1/F[t]
end

function logdetF(F::Array{T, 3}, t::Int) where T
    return @inbounds @views logdet(F[:, :, t])
end

function find_missing_observations(y::Matrix{T}) where T
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
    print("optimization_method = $(typeof(ss.opt_method)).")
    return nothing
end

function Base.show(io::IO, model::StateSpaceModel)
    println("A $(model.mode) state-space model with")
    println("n = $(model.dim.n),")
    println("p = $(model.dim.p),")
    println("m = $(model.dim.m),")
    print("r = $(model.dim.r).")
    return nothing
end

function build_H(p::Int, Typ)
    H = fill(NaN, p, p)
    return Typ.(H)
end

function build_Q(r::Int, p::Int, Typ)
    Q = kron(ones(Typ, p, p), Matrix{Typ}(I, Int(r/p), Int(r/p)))
    Q[findall(isequal(1), Q)] .= NaN
    return Q
end

function ensure_is_matrix(y::Vector{T}) where T
    return y[:, :]
end
function ensure_is_matrix(y::Matrix{T}) where T
    return y
end

function build_ss_dim(y::Matrix{Typ}, Z::Array{Typ, 3}, T::Matrix{Typ}, R::Matrix{Typ}) where Typ
    ny, py = size(y)
    pz, mz, nz = size(Z)
    mt1, mt2 = size(T)
    mr, rr = size(R)
    if !((mz == mt1 == mt2 == mr) && (pz == py))
        error("StateSpaceModel dimension mismatch")
    end
    return StateSpaceDimensions(ny, py, mr, rr)
end

function build_ss_dim(y::Matrix{Typ}, Z::Matrix{Typ}, T::Matrix{Typ}, R::Matrix{Typ}) where Typ
    ny, py = size(y)
    pz, mz = size(Z)
    mt, mt = size(T)
    mr, rr = size(R)
    if !((mz == mt == mr) && (pz == py))
        error("StateSpaceModel dimension mismatch")
    end
    return StateSpaceDimensions(ny, py, mr, rr)
end

function has_unknowns(model::StateSpaceModel{T}) where T
    unknowns = Unknowns(model)
    return unknowns.n_unknowns == 0 ? false : true
end

"""
    assert_dimensions(c::Matrix{T}, d::Matrix{T}, dim::StateSpaceDimensions) where T

Ensure dimensions of `c` and `d` are coherent.
"""
function assert_dimensions(c::Matrix{T}, d::Matrix{T}, dim::StateSpaceDimensions) where T
    @assert size(d, 1) >= dim.n
    @assert size(d, 2) == dim.p
    @assert size(c, 1) >= dim.n
    @assert size(c, 2) == dim.m
end
