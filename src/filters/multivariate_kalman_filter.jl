mutable struct MultivariateKalmanState{Fl<:AbstractFloat}
    v::Vector{Fl}
    F::Matrix{Fl}
    att::Vector{Fl}
    a::Vector{Fl}
    Ptt::Matrix{Fl}
    P::Matrix{Fl}
    llk::Fl
    steady_state::Bool
    # Auxiliary matrices
    P_to_check_steady_state::Matrix{Fl}
    TPtt::Matrix{Fl}
    function MultivariateKalmanState(
        num_series::Int, a1::Vector{Fl}, P1::Matrix{Fl}
    ) where Fl
        m = length(a1)
        P_to_check_steady_state = zeros(Fl, m, m)
        TPtt = zeros(Fl, m, m)

        return new{Fl}(
            zeros(Fl, num_series),
            zeros(Fl, num_series, num_series),
            zeros(Fl, m),
            a1,
            zeros(Fl, m, m),
            P1,
            zero(Fl),
            false,
            P_to_check_steady_state,
            TPtt,
        )
    end
end

function save_a1_P1_in_filter_output!(
    filter_output::FilterOutput{Fl}, kalman_state::MultivariateKalmanState{Fl}
) where Fl
    filter_output.a[1] = deepcopy(kalman_state.a)
    filter_output.P[1] = deepcopy(kalman_state.P)
    return filter_output
end

function save_kalman_state_in_filter_output!(
    filter_output::FilterOutput{Fl}, kalman_state::MultivariateKalmanState{Fl}, t::Int
) where Fl
    filter_output.v[t] = copy(kalman_state.v)
    filter_output.F[t] = copy(kalman_state.F)
    filter_output.a[t + 1] = copy(kalman_state.a)
    filter_output.att[t] = copy(kalman_state.att)
    filter_output.P[t + 1] = copy(kalman_state.P)
    filter_output.Ptt[t] = copy(kalman_state.Ptt)
    # There is no diffuse part in UnivariateKalmanState
    filter_output.Pinf[t] = copy(fill(zero(Fl), size(kalman_state.Ptt)))
    return filter_output
end

"""
TODO
"""
mutable struct MultivariateKalmanFilter{Fl<:Real} <: KalmanFilter
    steadystate_tol::Fl
    a1::Vector{Fl}
    P1::Matrix{Fl}
    skip_llk_instants::Int
    kalman_state::MultivariateKalmanState

    function MultivariateKalmanFilter(
        num_series::Int,
        a1::Vector{Fl},
        P1::Matrix{Fl},
        skip_llk_instants::Int=length(a1),
        steadystate_tol::Fl=Fl(1e-5),
    ) where Fl
        kalman_state = MultivariateKalmanState(num_series, copy(a1), copy(P1))
        return new{Fl}(steadystate_tol, a1, P1, skip_llk_instants, kalman_state)
    end
end

function reset_filter!(kf::MultivariateKalmanFilter{Fl}) where Fl
    copyto!(kf.kalman_state.a, kf.a1)
    copyto!(kf.kalman_state.P, kf.P1)
    set_state_llk_to_zero!(kf.kalman_state)
    fill!(kf.kalman_state.P_to_check_steady_state, zero(Fl))
    fill!(kf.kalman_state.TPtt, zero(Fl))
    kf.kalman_state.steady_state = false
    return kf
end

function set_state_llk_to_zero!(kalman_state::MultivariateKalmanState{Fl}) where Fl
    kalman_state.llk = zero(Fl)
    return kalman_state
end

# TODO: this should either be ! or return something
function check_steady_state(kalman_state::MultivariateKalmanState{Fl}, tol::Fl) where Fl
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
    kalman_state::MultivariateKalmanState{Fl}, y::Vector{Fl}, Z::Matrix{Fl}, d::Vector{Fl}
) where Fl
    kalman_state.v .= y .- Z * kalman_state.a .- d
    return kalman_state
end

function update_F!(
    kalman_state::MultivariateKalmanState{Fl}, Z::Matrix{Fl}, H::Matrix{Fl}
) where Fl
    kalman_state.F = Z * kalman_state.P * Z' .+ H
    return kalman_state
end

function update_att!(kalman_state::MultivariateKalmanState{Fl}, Z::Matrix{Fl}) where Fl
    kalman_state.att =
        kalman_state.a + kalman_state.P * Z' * inv(kalman_state.F) * kalman_state.v
    return kalman_state
end

function repeat_a_in_att!(kalman_state::MultivariateKalmanState{Fl}) where Fl
    for i in eachindex(kalman_state.att)
        kalman_state.att[i] = kalman_state.a[i]
    end
    return kalman_state
end

function update_a!(
    kalman_state::MultivariateKalmanState{Fl}, T::Matrix{Fl}, c::Vector{Fl}
) where Fl
    LinearAlgebra.BLAS.gemv!('N', one(Fl), T, kalman_state.att, zero(Fl), kalman_state.a)
    kalman_state.a .+= c
    return kalman_state
end

function update_Ptt!(kalman_state::MultivariateKalmanState{Fl}, Z::Matrix{Fl}) where Fl
    kalman_state.Ptt =
        kalman_state.P - (kalman_state.P * Z' * inv(kalman_state.F) * Z * kalman_state.P)
    return kalman_state
end

function repeat_P_in_Ptt!(kalman_state::MultivariateKalmanState{Fl}) where Fl
    for j in axes(kalman_state.P, 1), i in axes(kalman_state.P, 2)
        kalman_state.Ptt[i, j] = kalman_state.P[i, j]
    end
    return kalman_state
end

function update_P!(
    kalman_state::MultivariateKalmanState{Fl}, T::Matrix{Fl}, RQR::Matrix{Fl}
) where Fl
    LinearAlgebra.BLAS.gemm!(
        'N', 'N', Fl(1.0), T, kalman_state.Ptt, Fl(0.0), kalman_state.TPtt
    )
    LinearAlgebra.BLAS.gemm!(
        'N', 'T', Fl(1.0), kalman_state.TPtt, T, Fl(0.0), kalman_state.P
    )
    kalman_state.P .+= RQR
    return kalman_state
end

function update_P!(
    kalman_state::MultivariateKalmanState{Fl}, T::Matrix{Fl}, R::Matrix{Fl}, Q::Matrix{Fl}
) where Fl
    LinearAlgebra.BLAS.gemm!(
        'N', 'N', Fl(1.0), T, kalman_state.Ptt, Fl(0.0), kalman_state.TPtt
    )
    LinearAlgebra.BLAS.gemm!(
        'N', 'T', Fl(1.0), kalman_state.TPtt, T, Fl(0.0), kalman_state.P
    )
    kalman_state.P .+= R * Q * R'
    return kalman_state
end

function update_llk!(kalman_state::MultivariateKalmanState{Fl}) where Fl
    kalman_state.llk -= (
        HALF_LOG_2_PI + 0.5 * (logdet(kalman_state.F) +
        kalman_state.v' * inv(kalman_state.F) * kalman_state.v)
    )
    return kalman_state
end

function update_kalman_state!(
    kalman_state::MultivariateKalmanState{Fl},
    y::Vector{Fl},
    Z::Matrix{Fl},
    T::Matrix{Fl},
    H::Matrix{Fl},
    RQR::Matrix{Fl},
    d::Vector{Fl},
    c::Vector{Fl},
    skip_llk_instants::Int,
    tol::Fl,
    t::Int,
) where Fl
    if hasnan(y)
        fill!(kalman_state.v, NaN)
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
        update_Ptt!(kalman_state, Z)
        update_P!(kalman_state, T, RQR)
        check_steady_state(kalman_state, tol)
        if t > skip_llk_instants
            update_llk!(kalman_state)
        end
    end
    return kalman_state
end

function update_kalman_state!(
    kalman_state::MultivariateKalmanState{Fl},
    y::Vector{Fl},
    Z::Matrix{Fl},
    T::Matrix{Fl},
    H::Matrix{Fl},
    R::Matrix{Fl},
    Q::Matrix{Fl},
    d::Vector{Fl},
    c::Vector{Fl},
    skip_llk_instants::Int,
    t::Int,
) where Fl
    if kalman_state.steady_state
        update_v!(kalman_state, y, Z, d)
        update_att!(kalman_state, Z)
        update_a!(kalman_state, T, c)
    elseif hasnan(y)
        fill!(kalman_state.v, NaN)
        update_F!(kalman_state, Z, H)
        repeat_a_in_att!(kalman_state)
        update_a!(kalman_state, T, c)
        repeat_P_in_Ptt!(kalman_state)
        update_P!(kalman_state, T, R, Q)
    else
        update_v!(kalman_state, y, Z, d)
        update_F!(kalman_state, Z, H)
        update_att!(kalman_state, Z)
        update_a!(kalman_state, T, c)
        update_Ptt!(kalman_state, Z)
        update_P!(kalman_state, T, R, Q)
    end
    if t > skip_llk_instants
        update_llk!(kalman_state)
    end
    return kalman_state
end

function optim_kalman_filter(
    sys::StateSpaceSystem, filter::MultivariateKalmanFilter{Fl}
) where Fl
    return filter_recursions!(
        filter.kalman_state, sys, filter.steadystate_tol, filter.skip_llk_instants
    )
end

function kalman_filter!(
    filter_output::FilterOutput, sys::StateSpaceSystem, filter::MultivariateKalmanFilter{Fl}
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
    kalman_state::MultivariateKalmanState{Fl},
    sys::LinearMultivariateTimeInvariant,
    steadystate_tol::Fl,
    skip_llk_instants::Int,
) where Fl
    try
        RQR = sys.R * sys.Q * sys.R'
        @inbounds for t in 1:size(sys.y, 1)
            update_kalman_state!(
                kalman_state,
                sys.y[t, :],
                sys.Z,
                sys.T,
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

# TODO this doesn't use steadystate_tol
function filter_recursions!(
    kalman_state::MultivariateKalmanState{Fl},
    sys::LinearMultivariateTimeVariant,
    steadystate_tol::Fl,
    skip_llk_instants::Int,
) where Fl
    try
        @inbounds for t in 1:size(sys.y, 1)
            update_kalman_state!(
                kalman_state,
                sys.y[t, :],
                sys.Z[t],
                sys.T[t],
                sys.H[t],
                sys.R[t],
                sys.Q[t],
                sys.d[t],
                sys.c[t],
                skip_llk_instants,
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
    kalman_state::MultivariateKalmanState{Fl},
    sys::LinearMultivariateTimeInvariant,
    steadystate_tol::Fl,
    skip_llk_instants::Int,
) where Fl
    try
        RQR = sys.R * sys.Q * sys.R'
        save_a1_P1_in_filter_output!(filter_output, kalman_state)
        @inbounds for t in 1:size(sys.y, 1)
            update_kalman_state!(
                kalman_state,
                sys.y[t, :],
                sys.Z,
                sys.T,
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

function filter_recursions!(
    filter_output::FilterOutput,
    kalman_state::MultivariateKalmanState{Fl},
    sys::LinearMultivariateTimeVariant,
    steadystate_tol::Fl,
    skip_llk_instants::Int,
) where Fl
    try 
        save_a1_P1_in_filter_output!(filter_output, kalman_state)
        @inbounds for t in 1:size(sys.y, 1)
            update_kalman_state!(
                kalman_state,
                sys.y[t, :],
                sys.Z[t],
                sys.T[t],
                sys.H[t],
                sys.R[t],
                sys.Q[t],
                sys.d[t],
                sys.c[t],
                skip_llk_instants,
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
