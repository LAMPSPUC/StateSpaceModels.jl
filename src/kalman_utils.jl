# Utils for filter performance
function fill_a1(a::Matrix{T}) where T <: AbstractFloat
    for i in axes(a, 2)
        a[1, i] = zero(T)
    end
end

function fill_P1(P::AbstractArray{T}; bigkappa::Float64 = 1e6) where T <: AbstractFloat
    for i in axes(P, 1), j in axes(P, 2)
        if i == j
            P[i, j, 1] = bigkappa
        else
            P[i, j, 1] = zero(T)
        end
    end
end

function repeat_matrix_t_plus_1(mat::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @inbounds for j in axes(mat, 2), i in axes(mat, 1)
          mat[i, j, t+1] = mat[i, j, t]
    end
    return 
end

function update_a(a::Matrix{Typ}, att::Matrix{Typ}, T::Matrix{Typ}, t::Int) where Typ <: AbstractFloat
    @views @inbounds mul!(a[t+1, :], T, att[t, :])
    return 
end

function update_ZP(ZP::AbstractArray{T}, Z::AbstractArray{T}, P::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @inbounds @views mul!(ZP, Z[:, :, t], P[:, :, t])
    return 
end

function update_K(K::AbstractArray{Typ}, P_Ztransp_invF::AbstractArray{Typ}, T::AbstractArray{Typ}, t::Int) where Typ <: AbstractFloat
    @inbounds @views mul!(K[:, :, t], T, P_Ztransp_invF)
    return
end

function invF(F::AbstractArray{T}, t::Int) where T
    return @inbounds @views invertF(F[:, :, t]) 
end

function update_P(P::AbstractArray{Typ}, T::AbstractArray{Typ}, Ptt::AbstractArray{Typ}, RQR::AbstractArray{Typ}, t::Int)  where Typ <: AbstractFloat 
    @views @inbounds LinearAlgebra.BLAS.gemm!('N', 'T', 1.0, T * Ptt[:, :, t], T, 0.0, P[:, :, t+1]) # P[:, :, t+1] = T * Ptt[:, :, t] * T'
    sum_matrix!(P, RQR, t, 1) # P[:, :, t+1] .+= RQR
    ensure_pos_sym!(P, t + 1)
    return 
end

function update_v(v::AbstractArray{T}, y::AbstractArray{T}, Z::AbstractArray{T}, a::AbstractArray{T}, t::Int) where T <: AbstractFloat
    # v[t, :] = y[t, :] - Z[:, :, t]*a[t, :]
    for i in axes(Z, 1)
        v[t, i] = y[t, i]
        for j in axes(Z, 2)
            v[t, i] -= Z[i, j, t]*a[t, j]
        end
    end
    return 
end

function update_att(att::AbstractArray{T}, a::AbstractArray{T}, P_Ztransp_invF::AbstractArray{T}, v::AbstractArray{T}, t::Int) where T <: AbstractFloat
    # att[t, :] = a[t, :] + P_Ztransp_invF*v[t, :]
    for i in axes(P_Ztransp_invF, 1)
        att[t, i] = a[t, i]
        for j in axes(P_Ztransp_invF, 2)
            att[t, i] += P_Ztransp_invF[i,j]*v[t, j]
        end
    end
    return 
end

function update_att(att::AbstractArray{T}, a::AbstractArray{T}, P::AbstractArray{T}, Z::AbstractArray{T},
                    F::AbstractArray{T}, v::AbstractArray{T}, t::Int) where T <: AbstractFloat

    @views @inbounds att[t, :] = a[t, :] + 
        LinearAlgebra.BLAS.gemm('N', 'T', 1.0, P[:, :, t], Z[:, :, t]) *
        invF(F, t) * v[t, :]
    return
end

function update_att(att::AbstractArray{T}, a::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @inbounds for j in axes(att, 2)
        att[t, j] = a[t, j]
    end
    return
end

function update_Ptt(Ptt::AbstractArray{T}, P::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @inbounds for i in axes(Ptt, 1), j in axes(Ptt, 2)
        Ptt[i, j, t] = P[i, j, t]
    end
    return
end

function update_Ptt(Ptt::AbstractArray{T}, P::AbstractArray{T}, P_Ztransp_invF::AbstractArray{T},
                    ZP::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @views @inbounds mul!(Ptt[:, :, t], -P_Ztransp_invF, ZP) # Ptt_t = - P_Ztransp_invF * ZP
    sum_matrix!(Ptt, P, t, 0) # Ptt_t += P_t
    return 
end

function update_F(F::AbstractArray{T}, ZP::AbstractArray{T}, Z::AbstractArray{T}, H::AbstractArray{T}, t::Int) where T <: AbstractFloat
    @views @inbounds LinearAlgebra.BLAS.gemm!('N', 'T', 1.0, ZP, Z[:, :, t], 0.0, F[:, :, t]) # F_t = Z_t * P_t * Z_t
    sum_matrix!(F, H, t, 0) # F[:, :, t] .+= H
    return
end

function update_P_Ztransp_Finv(P_Ztransp_invF::AbstractArray{T}, ZP::AbstractArray{T}, F::AbstractArray{T}, t::Int) where T <: AbstractFloat
    LinearAlgebra.BLAS.gemm!('T', 'N', 1.0, ZP, invF(F, t), 0.0, P_Ztransp_invF)
    return
end