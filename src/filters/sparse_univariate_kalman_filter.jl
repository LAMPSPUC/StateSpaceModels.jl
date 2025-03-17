mutable struct SparseUnivariateKalmanState{Fl<:AbstractFloat}
    v::Fl
    F::Fl
    att::Vector{Fl}
    a::Vector{Fl}
    Ptt::Matrix{Fl}
    P::Matrix{Fl}
    llk::Fl
    steady_state::Bool
    # Auxiliary matrices
    P_to_check_steady_state::Matrix{Fl}
    ZP::Vector{Fl}
    TPtt::Matrix{Fl}
    function SparseUnivariateKalmanState(a1::Vector{Fl}, P1::Matrix{Fl}) where Fl
        m = length(a1)
        P_to_check_steady_state = zeros(Fl, m, m)
        ZP = zeros(Fl, m)
        TPtt = zeros(Fl, m, m)

        return new{Fl}(
            zero(Fl),
            zero(Fl),
            zeros(Fl, m),
            a1,
            zeros(Fl, m, m),
            P1,
            zero(Fl),
            false,
            P_to_check_steady_state,
            ZP,
            TPtt,
        )
    end
end

function save_a1_P1_in_filter_output!(
    filter_output::FilterOutput{Fl}, kalman_state::SparseUnivariateKalmanState{Fl}
) where Fl
    filter_output.a[1] = deepcopy(kalman_state.a)
    filter_output.P[1] = deepcopy(kalman_state.P)
    return filter_output
end

function save_kalman_state_in_filter_output!(
    filter_output::FilterOutput{Fl}, kalman_state::SparseUnivariateKalmanState{Fl}, t::Int
) where Fl
    filter_output.v[t] = copy(fill(kalman_state.v, 1))
    filter_output.F[t] = copy(fill(kalman_state.F, 1, 1))
    filter_output.a[t + 1] = copy(kalman_state.a)
    filter_output.att[t] = copy(kalman_state.att)
    filter_output.P[t + 1] = copy(kalman_state.P)
    filter_output.Ptt[t] = copy(kalman_state.Ptt)
    # There is no diffuse part in SparseUnivariateKalmanState
    filter_output.Pinf[t] = copy(fill(zero(Fl), size(kalman_state.Ptt)))
    return filter_output
end

# Univariate Kalman filter with the recursions as described
# in Koopman's book TODO
"""
    SparseUnivariateKalmanFilter{Fl <: AbstractFloat}

A Kalman filter that is tailored to sparse univariate systems, exploiting the fact that the
dimension of the observations at any time period is 1 and that Z, T and R are sparse.

# TODO equations and descriptions of a1 and P1
"""
mutable struct SparseUnivariateKalmanFilter{Fl<:AbstractFloat} <: KalmanFilter
    steadystate_tol::Fl
    a1::Vector{Fl}
    P1::Matrix{Fl}
    skip_llk_instants::Int
    kalman_state::SparseUnivariateKalmanState

    function SparseUnivariateKalmanFilter(
        a1::Vector{Fl},
        P1::Matrix{Fl},
        skip_llk_instants::Int=length(a1),
        steadystate_tol::Fl=Fl(1e-5),
    ) where Fl
        kalman_state = SparseUnivariateKalmanState(copy(a1), copy(P1))
        return new{Fl}(steadystate_tol, a1, P1, skip_llk_instants, kalman_state)
    end
end

function reset_filter!(kf::SparseUnivariateKalmanFilter{Fl}) where Fl
    copyto!(kf.kalman_state.a, kf.a1)
    copyto!(kf.kalman_state.P, kf.P1)
    set_state_llk_to_zero!(kf.kalman_state)
    fill!(kf.kalman_state.P_to_check_steady_state, zero(Fl))
    fill!(kf.kalman_state.ZP, zero(Fl))
    fill!(kf.kalman_state.TPtt, zero(Fl))
    kf.kalman_state.steady_state = false
    return kf
end

function set_state_llk_to_zero!(kalman_state::SparseUnivariateKalmanState{Fl}) where Fl
    kalman_state.llk = zero(Fl)
    return kalman_state
end

# TODO this should either have ! or return something
function check_steady_state(kalman_state::SparseUnivariateKalmanState{Fl}, tol::Fl) where Fl
    @inbounds for j in axes(kalman_state.P, 2), i in axes(kalman_state.P, 1)
        if abs(
            (kalman_state.P[i, j] - kalman_state.P_to_check_steady_state[i, j]) /
            kalman_state.P[i, j],
        ) > tol
            # Update the P_to_check_steady_state matrix
            copyto!(kalman_state.P_to_check_steady_state, kalman_state.P)
            return nothing
        end
    end
    kalman_state.steady_state = true
    return nothing
end

function update_v!(
    kalman_state::SparseUnivariateKalmanState{Fl}, y::Fl, Z::SparseVector{Fl, Int}, d::Fl
) where Fl
    kalman_state.v = y - dot(Z, kalman_state.a) - d
    return kalman_state
end

function update_F!(kalman_state::SparseUnivariateKalmanState{Fl}, Z::SparseVector{Fl, Int}, H::Fl) where Fl
    kalman_state.ZP = kalman_state.P * Z
    kalman_state.F = H
    kalman_state.F += dot(kalman_state.ZP, Z)
    return kalman_state
end

function update_att!(kalman_state::SparseUnivariateKalmanState{Fl}, Z::SparseVector{Fl, Int}) where Fl
    copyto!(kalman_state.att, kalman_state.a)
    # Here we can simplify the j iterator # TODO
    @inbounds for i in axes(kalman_state.P, 1), j in axes(kalman_state.P, 2)
        kalman_state.att[i] +=
            (kalman_state.v / kalman_state.F) * kalman_state.P[i, j] * Z[j]
    end
    return kalman_state
end

function repeat_a_in_att!(kalman_state::SparseUnivariateKalmanState{Fl}) where Fl
    for i in eachindex(kalman_state.att)
        kalman_state.att[i] = kalman_state.a[i]
    end
    return kalman_state
end

function update_a!(
    kalman_state::SparseUnivariateKalmanState{Fl}, T::SparseMatrixCSC{Fl, Int}, c::Vector{Fl}
) where Fl
    kalman_state.a = T * kalman_state.att
    kalman_state.a .+= c
    return kalman_state
end

function update_Ptt!(kalman_state::SparseUnivariateKalmanState{Fl}) where Fl
    LinearAlgebra.BLAS.gemm!(
        'N', 'T', one(Fl), kalman_state.ZP, kalman_state.ZP, zero(Fl), kalman_state.Ptt
    )
    @. kalman_state.Ptt = kalman_state.P - (kalman_state.Ptt / kalman_state.F)
    return kalman_state
end

function repeat_P_in_Ptt!(kalman_state::SparseUnivariateKalmanState{Fl}) where Fl
    for j in axes(kalman_state.P, 1), i in axes(kalman_state.P, 2)
        kalman_state.Ptt[i, j] = kalman_state.P[i, j]
    end
    return kalman_state
end

function update_P!(
    kalman_state::SparseUnivariateKalmanState{Fl}, T::SparseMatrixCSC{Fl, Int}, RQR::Matrix{Fl}
) where Fl
    mul!(kalman_state.TPtt, T, kalman_state.Ptt)
    mul!(kalman_state.P, kalman_state.TPtt, T')
    kalman_state.P .+= RQR
    return kalman_state
end

function update_llk!(kalman_state::SparseUnivariateKalmanState{Fl}) where Fl
    kalman_state.llk -= (
        HALF_LOG_2_PI + 0.5 * (log(kalman_state.F) + kalman_state.v^2 / kalman_state.F)
    )
    return kalman_state
end

function update_kalman_state!(
    kalman_state::SparseUnivariateKalmanState{Fl},
    y::Fl,
    Z::SparseVector{Fl, Int},
    T::SparseMatrixCSC{Fl, Int},
    H::Fl,
    RQR::Matrix{Fl},
    d::Fl,
    c::Vector{Fl},
    skip_llk_instants::Int,
    tol::Fl,
    t::Int,
) where Fl
    if isnan(y)
        kalman_state.v = NaN
        update_F!(kalman_state, Z, H)
        repeat_a_in_att!(kalman_state)
        update_a!(kalman_state, T, c)
        repeat_P_in_Ptt!(kalman_state)
        update_P!(kalman_state, T, RQR)
        kalman_state.steady_state = false # Not on steadystate anymore
    elseif kalman_state.steady_state
        update_v!(kalman_state, y, Z, d)
        update_att!(kalman_state, Z)
        update_a!(kalman_state, T, c)
        if t > skip_llk_instants
            update_llk!(kalman_state)
        end
    else
        update_v!(kalman_state, y, Z, d)
        update_F!(kalman_state, Z, H)
        update_att!(kalman_state, Z)
        update_a!(kalman_state, T, c)
        update_Ptt!(kalman_state)
        update_P!(kalman_state, T, RQR)
        check_steady_state(kalman_state, tol)
        if t > skip_llk_instants
            update_llk!(kalman_state)
        end
    end
    return kalman_state
end

function optim_kalman_filter(
    sys::StateSpaceSystem, filter::SparseUnivariateKalmanFilter{Fl}
) where Fl
    return filter_recursions!(
        filter.kalman_state, sys, filter.steadystate_tol, filter.skip_llk_instants
    )
end

function kalman_filter!(
    filter_output::FilterOutput, sys::StateSpaceSystem, filter::SparseUnivariateKalmanFilter{Fl}
) where Fl
    filter_recursions!(
        filter_output,
        filter.kalman_state,
        sys,
        filter.steadystate_tol,
        filter.skip_llk_instants,
    )
    return filter_output
end

function filter_recursions!(
    kalman_state::SparseUnivariateKalmanState{Fl},
    sys::LinearUnivariateTimeInvariant,
    steadystate_tol::Fl,
    skip_llk_instants::Int,
) where Fl
    try 
        RQR = sys.R * sys.Q * sys.R'
        T_sparse = sparse(sys.T) 
        Z_sparse = sparse(sys.Z)
        @inbounds for t in eachindex(sys.y)
            update_kalman_state!(
                kalman_state,
                sys.y[t],
                Z_sparse,
                T_sparse,
                sys.H,
                RQR,
                sys.d,
                sys.c,
                skip_llk_instants,
                steadystate_tol,
                t,
            )
        end
    catch
        @error("Numerical error when applying Kalman filter equations")
        rethrow()
    end
    return kalman_state.llk
end

function filter_recursions!(
    filter_output::FilterOutput,
    kalman_state::SparseUnivariateKalmanState{Fl},
    sys::LinearUnivariateTimeInvariant,
    steadystate_tol::Fl,
    skip_llk_instants::Int,
) where Fl
    try
        RQR = sys.R * sys.Q * sys.R'
        T_sparse = sparse(sys.T) 
        Z_sparse = sparse(sys.Z)
        save_a1_P1_in_filter_output!(filter_output, kalman_state)
        @inbounds for t in eachindex(sys.y)
            update_kalman_state!(
                kalman_state,
                sys.y[t],
                Z_sparse,
                T_sparse,
                sys.H,
                RQR,
                sys.d,
                sys.c,
                skip_llk_instants,
                steadystate_tol,
                t,
            )
            save_kalman_state_in_filter_output!(filter_output, kalman_state, t)
        end
    catch
        @error("Numerical error when applying Kalman filter equations")
        rethrow()
    end
    return filter_output
end
