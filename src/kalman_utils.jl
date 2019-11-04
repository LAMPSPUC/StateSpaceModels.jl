# Utils for filter performance
function fill_a1!(a::Matrix{T}) where T
    @inbounds for i in axes(a, 2)
        a[1, i] = zero(T)
    end
end
function fill_P1!(P::AbstractArray{T}; bigkappa::T = 1e6) where T
    @inbounds for i in axes(P, 1), j in axes(P, 2)
        if i == j
            P[i, j, 1] = bigkappa
        else
            P[i, j, 1] = zero(T)
        end
    end
end

function repeat_matrix_t_plus_1!(mat::AbstractArray{T}, t::Int) where T
    @inbounds for j in axes(mat, 2), i in axes(mat, 1)
          mat[i, j, t+1] = mat[i, j, t]
    end
    return 
end
function repeat_vector_t_plus_1!(mat::Matrix{T}, t::Int) where T
    @inbounds for i in axes(mat, 1)
        mat[i, t+1] = mat[i, t]
    end
    return 
end

function big_update_a!(a::Matrix{Typ}, att::Matrix{Typ}, T::Matrix{Typ}, c::Matrix{Typ}, t::Int) where Typ
    @views @inbounds mul!(a[t+1, :], T, att[t, :])
    for i in axes(a, 2)
        a[t + 1, i] += c[t, i]
    end
    return 
end
function small_update_a!(a::Matrix{Typ}, att::Matrix{Typ}, T::Matrix{Typ}, c::Matrix{Typ}, t::Int) where Typ
    @inbounds for i in axes(a, 2)
        a[t+1, i] = c[t, i]
        for j in axes(T, 2)
            a[t+1, i] += T[i, j]*att[t, j]
        end
    end
    return 
end
function update_a!(a::Matrix{Typ}, att::Matrix{Typ}, T::Matrix{Typ}, c::Matrix{Typ}, t::Int) where Typ
    # Here there is a trade-off between memory and speed, usually if the dimension of a is smaller than 15 
    # it is more performant to do the hard-coded version (small_update_a)
    if size(a, 2) < 16
        small_update_a!(a, att, T, c, t)
    else
        big_update_a!(a, att, T, c, t)
    end
    return 
end

function update_ZP!(ZP::Matrix{T}, Z::Array{T, 3}, P::Array{T, 3}, t::Int) where T
    @inbounds @views mul!(ZP, Z[:, :, t], P[:, :, t])
    return 
end
function update_ZP!(ZP::Vector{T}, Z::Array{T, 3}, P::Array{T, 3}, t::Int) where T
    @inbounds for i in axes(ZP, 1)
        ZP[i] = zero(T)
        for j in axes(ZP, 1)
            ZP[i] += Z[1, j, t]*P[j, i, t]
        end
    end
    return 
end

function update_K!(K::AbstractArray{Typ}, P_Ztransp_invF::AbstractMatrix{Typ}, T::AbstractMatrix{Typ}, t::Int) where Typ
    @inbounds @views mul!(K[:, :, t], T, P_Ztransp_invF)
    return
end

function update_K!(K::AbstractMatrix{Typ}, P_Ztransp_invF::AbstractVector{Typ}, T::AbstractMatrix{Typ}, t::Int) where Typ
    # Here there is a trade-off memory-speed, usually if the dimension of a is smaller than 15 
    # it is more performant to do the hard coded version (small_update_K!)
    if size(K, 1) < 16
        small_update_K!(K, P_Ztransp_invF, T, t)
    else
        big_update_K!(K, P_Ztransp_invF, T, t)
    end
    return 
end

function small_update_K!(K::AbstractMatrix{Typ}, P_Ztransp_invF::AbstractVector{Typ}, T::AbstractMatrix{Typ}, t::Int) where Typ
    @inbounds for i in axes(K, 1)
        K[i, t] = zero(Typ) 
        for j in axes(K, 1)
            K[i, t] += T[i, j]*P_Ztransp_invF[j]
        end
    end
    return
end

function big_update_K!(K::AbstractMatrix{Typ}, P_Ztransp_invF::AbstractVector{Typ}, T::AbstractMatrix{Typ}, t::Int) where Typ
    @inbounds @views K[:, t] = T * P_Ztransp_invF
    return
end

function invF(F::AbstractArray{T}, t::Int) where T
    return @inbounds @views invertF(F, t) 
end

function update_P!(P::Array{Typ, 3}, T::Matrix{Typ}, Ptt::Array{Typ, 3}, RQR::Matrix{Typ}, t::Int)  where Typ
    @views @inbounds LinearAlgebra.BLAS.gemm!('N', 'T', Typ(1.0), T * Ptt[:, :, t], T, Typ(0.0), P[:, :, t+1]) # P[:, :, t+1] = T * Ptt[:, :, t] * T'
    sum_matrix!(P, RQR, t, 1) # P[:, :, t+1] .+= RQR
    ensure_pos_sym!(P, t + 1)
    return 
end

function update_v!(v::Matrix{T}, y::Matrix{T}, Z::Array{T, 3}, d::Matrix{T}, a::Matrix{T}, t::Int) where T
    # v[t, :] = y[t, :] - Z[:, :, t]*a[t, :]
    @inbounds for i in axes(Z, 1)
        v[t, i] = y[t, i] - d[t, i]
        for j in axes(Z, 2)
            v[t, i] -= Z[i, j, t]*a[t, j]
        end
    end
    return 
end

function update_v!(v::Vector{T}, y::Matrix{T}, Z::Array{T, 3}, d::Matrix{T}, a::Matrix{T}, t::Int) where T
    # v[t, :] = y[t, :] - Z[:, :, t]*a[t, :]
    v[t] = y[t, 1] - d[t, 1]
    @inbounds for j in axes(Z, 2)
        v[t] -= Z[1, j, t]*a[t, j]
    end
    return 
end

function update_att!(att::Matrix{T}, a::Matrix{T}, P_Ztransp_invF::Matrix{T}, v::Matrix{T}, t::Int) where T
    # att[t, :] = a[t, :] + P_Ztransp_invF*v[t, :]
    @inbounds for i in axes(P_Ztransp_invF, 1)
        att[t, i] = a[t, i]
        for j in axes(P_Ztransp_invF, 2)
            att[t, i] += P_Ztransp_invF[i, j]*v[t, j]
        end
    end
    return 
end

function update_att!(att::Matrix{T}, a::Matrix{T}, P_Ztransp_invF::Vector{T}, v::Vector{T}, t::Int) where T
    # att[t, :] = a[t, :] + P_Ztransp_invF*v[t, :]
    @inbounds for i in axes(P_Ztransp_invF, 1)
        att[t, i] = a[t, i] + P_Ztransp_invF[i]*v[t]
    end
    return 
end

function update_att!(att::Matrix{T}, a::Matrix{T}, t::Int) where T
    @inbounds for j in axes(att, 2)
        att[t, j] = a[t, j]
    end
    return
end

function update_Ptt!(Ptt::Array{T, 3}, P::Array{T, 3}, t::Int) where T
    @inbounds for i in axes(Ptt, 1), j in axes(Ptt, 2)
        Ptt[i, j, t] = P[i, j, t]
    end
    return
end

function update_Ptt!(Ptt::Array{T, 3}, P::Array{T, 3}, P_Ztransp_invF::Matrix{T},
                    ZP::Matrix{T}, t::Int) where T
    @views @inbounds mul!(Ptt[:, :, t], -P_Ztransp_invF, ZP) # Ptt_t = - P_Ztransp_invF * ZP
    sum_matrix!(Ptt, P, t, 0) # Ptt_t += P_t
    return 
end
function update_Ptt!(Ptt::Array{T, 3}, P::Array{T, 3}, P_Ztransp_invF::Vector{T},
                    ZP::Vector{T}, t::Int) where T
    @inbounds for i in axes(Ptt, 1), j in axes(Ptt, 2)
        Ptt[i, j, t] = P[i, j, t] - P_Ztransp_invF[i]*ZP[j] # Ptt_t = - P_Ztransp_invF * ZP
    end
    return 
end

function update_F!(F::Array{T, 3}, ZP::Matrix{T}, Z::Array{T, 3}, H::AbstractArray{T}, t::Int) where T
    @views @inbounds LinearAlgebra.BLAS.gemm!('N', 'T', T(1.0), ZP, Z[:, :, t], T(0.0), F[:, :, t]) # F_t = Z_t * P_t * Z_t
    sum_matrix!(F, H, t, 0) # F[:, :, t] .+= H
    return
end

function update_F!(F::Vector{T}, ZP::Vector{T}, Z::Array{T, 3}, H::Matrix{T}, t::Int) where T
    F[t] = H[1, 1]
    @inbounds for i in axes(ZP, 1)
        F[t] += ZP[i]*Z[1, i, t] # F_t = Z_t * P_t * Z_t
    end
    return
end

function update_P_Ztransp_Finv!(P_Ztransp_invF::Matrix{T}, ZP::Matrix{T}, F::Array{T, 3}, t::Int) where T
    # P_Ztransp_invF = (ZP)' * F^-1
    if size(F, 1) == 1
        for p in axes(ZP, 1), m in axes(ZP, 2)
            P_Ztransp_invF[m, p] = ZP[p, m]/F[1, 1, t]
        end
    else
        LinearAlgebra.BLAS.gemm!('T', 'N', T(1.0), ZP, invF(F, t), T(0.0), P_Ztransp_invF)
    end
    return
end

function update_P_Ztransp_Finv!(P_Ztransp_invF::Vector{T}, ZP::Vector{T}, F::Vector{T}, t::Int) where T
    @inbounds for i in axes(P_Ztransp_invF, 1)
        P_Ztransp_invF[i] = ZP[i]/F[t]
    end
    return
end