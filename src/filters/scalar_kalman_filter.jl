mutable struct ScalarKalmanState{Fl<:AbstractFloat}
    v::Fl
    F::Fl
    att::Fl
    a::Fl
    Ptt::Fl
    P::Fl
    llk::Fl
    steady_state::Bool
    # Auxiliary matrices
    P_to_check_steady_state::Fl
    function ScalarKalmanState(a1::Fl, P1::Fl) where Fl
        P_to_check_steady_state = zero(Fl)
        return new{Fl}(
            zero(Fl),
            zero(Fl),
            zero(Fl),
            a1,
            zero(Fl),
            P1,
            zero(Fl),
            false,
            P_to_check_steady_state,
        )
    end
end

function save_a1_P1_in_filter_output!(
    filter_output::FilterOutput{Fl}, kalman_state::ScalarKalmanState{Fl}
) where Fl
    filter_output.a[1] = copy(fill(kalman_state.a, 1))
    filter_output.P[1] = copy(fill(kalman_state.P, 1, 1))
    return filter_output
end

function save_kalman_state_in_filter_output!(
    filter_output::FilterOutput{Fl}, kalman_state::ScalarKalmanState{Fl}, t::Int
) where Fl
    filter_output.v[t] = copy(fill(kalman_state.v, 1))
    filter_output.F[t] = copy(fill(kalman_state.F, 1, 1))
    filter_output.a[t + 1] = copy(fill(kalman_state.a, 1))
    filter_output.att[t] = copy(fill(kalman_state.att, 1))
    filter_output.P[t + 1] = copy(fill(kalman_state.P, 1, 1))
    filter_output.Ptt[t] = copy(fill(kalman_state.Ptt, 1, 1))
    # There is no diffuse part in UnivariateKalmanState
    filter_output.Pinf[t] = copy(fill(zero(Fl), size(filter_output.Ptt[t])))
    return filter_output
end

"""
    ScalarKalmanFilter{Fl <: Real} <: KalmanFilter

Similar to the univariate Kalman filter but exploits the fact that the dimension of the
state is equal to 1.
"""
mutable struct ScalarKalmanFilter{Fl<:Real} <: KalmanFilter
    steadystate_tol::Fl
    a1::Fl
    P1::Fl
    skip_llk_instants::Int
    kalman_state::ScalarKalmanState

    function ScalarKalmanFilter(
        a1::Fl, P1::Fl, skip_llk_instants::Int=1, steadystate_tol::Fl=Fl(1e-5)
    ) where Fl
        kalman_state = ScalarKalmanState(a1, P1)
        return new{Fl}(steadystate_tol, a1, P1, skip_llk_instants, kalman_state)
    end
end

function reset_filter!(kf::ScalarKalmanFilter{Fl}) where Fl
    reset_filter!(kf.kalman_state, kf.a1, kf.P1)
    return kf
end

function reset_filter!(kalman_state::ScalarKalmanState, a1::Fl, P1::Fl) where Fl
    kalman_state.a = a1
    kalman_state.P = P1
    kalman_state.llk = zero(Fl)
    kalman_state.P_to_check_steady_state = zero(Fl)
    kalman_state.steady_state = false
    return kalman_state
end

function scalar_check_steady_state(kalman_state::ScalarKalmanState{Fl}, tol::Fl) where Fl
    if abs((kalman_state.P - kalman_state.P_to_check_steady_state) / kalman_state.P) > tol
        kalman_state.P_to_check_steady_state = kalman_state.P
        return kalman_state
    end
    kalman_state.steady_state = true
    return kalman_state
end

function scalar_update_v!(
    kalman_state::ScalarKalmanState{Fl}, y::Fl, Z::Fl, d::Fl
) where Fl
    kalman_state.v = y - Z * kalman_state.a - d
    return kalman_state
end

function scalar_update_F!(kalman_state::ScalarKalmanState{Fl}, Z::Fl, H::Fl) where Fl
    kalman_state.F = Z * kalman_state.P * Z + H
    return kalman_state
end

function scalar_update_att!(kalman_state::ScalarKalmanState{Fl}, Z::Fl) where Fl
    kalman_state.att = kalman_state.a + Z * kalman_state.P * kalman_state.v / kalman_state.F
    return kalman_state
end

function scalar_update_a!(kalman_state::ScalarKalmanState{Fl}, T::Fl, c::Fl) where Fl
    kalman_state.a = T * kalman_state.att + c
    return kalman_state
end

function scalar_update_Ptt!(kalman_state::ScalarKalmanState{Fl}, Z::Fl) where Fl
    kalman_state.Ptt =
        kalman_state.P - kalman_state.P * Z * Z * kalman_state.P / kalman_state.F
    return kalman_state
end

function scalar_update_P!(kalman_state::ScalarKalmanState{Fl}, T::Fl, RQR::Fl) where Fl
    kalman_state.P = T * kalman_state.Ptt * T + RQR
    return kalman_state
end

function update_llk!(kalman_state::ScalarKalmanState{Fl}) where Fl
    kalman_state.llk -= (
        HALF_LOG_2_PI + 0.5 * (log(kalman_state.F) + kalman_state.v^2 / kalman_state.F)
    )
    return kalman_state
end

function scalar_update_kalman_state!(
    kalman_state, y, Z, T, H, RQR, d, c, skip_llk_instants, tol, t
)
    if isnan(y)
        kalman_state.v = NaN
        scalar_update_F!(kalman_state, Z, H)
        kalman_state.att = kalman_state.a
        scalar_update_a!(kalman_state, T, c)
        kalman_state.Ptt = kalman_state.P
        scalar_update_P!(kalman_state, T, RQR)
        kalman_state.steady_state = false # Not on steadystate anymore
    elseif kalman_state.steady_state
        scalar_update_v!(kalman_state, y, Z, d)
        scalar_update_att!(kalman_state, Z)
        scalar_update_a!(kalman_state, T, c)
        if t > skip_llk_instants
            update_llk!(kalman_state)
        end
    else
        scalar_update_v!(kalman_state, y, Z, d)
        scalar_update_F!(kalman_state, Z, H)
        scalar_update_att!(kalman_state, Z)
        scalar_update_a!(kalman_state, T, c)
        scalar_update_Ptt!(kalman_state, Z)
        scalar_update_P!(kalman_state, T, RQR)
        scalar_check_steady_state(kalman_state, tol)
        if t > skip_llk_instants
            update_llk!(kalman_state)
        end
    end
    return kalman_state
end

function optim_kalman_filter(
    sys::LinearUnivariateTimeInvariant{Fl}, filter::ScalarKalmanFilter{Fl}
) where Fl
    return filter_recursions!(
        filter.kalman_state, sys, filter.steadystate_tol, filter.skip_llk_instants
    )
end

function kalman_filter!(
    filter_output::FilterOutput, sys::StateSpaceSystem, filter::ScalarKalmanFilter{Fl}
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
    kalman_state::ScalarKalmanState{Fl},
    sys::LinearUnivariateTimeInvariant{Fl},
    steadystate_tol::Fl,
    skip_llk_instants::Int,
) where Fl
    try 
        RQR = sys.R[1] * sys.Q[1] * sys.R[1]
        @inbounds for t in eachindex(sys.y)
            scalar_update_kalman_state!(
                kalman_state,
                sys.y[t],
                sys.Z[1],
                sys.T[1],
                sys.H,
                RQR,
                sys.d,
                sys.c[1],
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
    kalman_state::ScalarKalmanState{Fl},
    sys::LinearUnivariateTimeInvariant,
    steadystate_tol::Fl,
    skip_llk_instants::Int,
) where Fl
    try
        RQR = sys.R[1] * sys.Q[1] * sys.R[1]
        save_a1_P1_in_filter_output!(filter_output, kalman_state)
        @inbounds for t in eachindex(sys.y)
            scalar_update_kalman_state!(
                kalman_state,
                sys.y[t],
                sys.Z[1],
                sys.T[1],
                sys.H,
                RQR,
                sys.d,
                sys.c[1],
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
