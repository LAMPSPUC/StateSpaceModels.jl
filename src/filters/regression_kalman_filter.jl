#     RegressionKalmanState{Fl <: AbstractFloat}

# Similar to the univariate Kalman filter, but exploits the structure of the regression system
# matrices - in particular, the fact that the variance of the state (`P`) is always zero, as
# well as `R` and `Q`.
mutable struct RegressionKalmanState{Fl<:AbstractFloat}
    v::Fl
    F::Fl
    a::Vector{Fl}
    llk::Fl

    function RegressionKalmanState(a1::Vector{Fl}) where Fl
        return new{Fl}(zero(Fl), zero(Fl), a1, zero(Fl))
    end
end

function save_kalman_state_in_filter_output!(
    filter_output::FilterOutput{Fl}, kalman_state::RegressionKalmanState{Fl}
) where Fl
    filter_output.a[1] = copy(kalman_state.a)
    filter_output.P[1] = copy(fill(zero(Fl), 1, 1))
    for t in 1:length(filter_output.v)
        filter_output.v[t] = copy(fill(kalman_state.v, 1))
        filter_output.F[t] = copy(fill(kalman_state.F, 1, 1))
        filter_output.a[t + 1] = copy(kalman_state.a)
        filter_output.att[t] = copy(kalman_state.a)
        filter_output.P[t + 1] = copy(fill(zero(Fl), 1, 1))
        filter_output.Ptt[t] = copy(fill(zero(Fl), 1, 1))
        # There is no diffuse part in UnivariateKalmanState
        filter_output.Pinf[t] = copy(fill(zero(Fl), size(filter_output.Ptt[t])))
    end
    return filter_output
end

mutable struct RegressionKalmanFilter{Fl<:AbstractFloat} <: KalmanFilter
    a1::Vector{Fl}
    kalman_state::RegressionKalmanState

    function RegressionKalmanFilter(a1::Vector{Fl}) where Fl
        kalman_state = RegressionKalmanState(a1)
        return new{Fl}(a1, kalman_state)
    end
end

function reset_filter!(filter::RegressionKalmanFilter{Fl}) where Fl
    copyto!(filter.kalman_state.a, filter.a1)
    filter.kalman_state.llk = zero(Fl)
    return filter
end

function regression_update_v!(
    kalman_state::RegressionKalmanState{Fl}, y::Fl, Z::Vector{Fl}
) where Fl
    kalman_state.v = y - dot(Z, kalman_state.a)
    return kalman_state
end

function update_llk!(kalman_state::RegressionKalmanState{Fl}) where Fl
    kalman_state.llk -= (
        HALF_LOG_2_PI + 0.5 * (log(kalman_state.F) + kalman_state.v^2 / kalman_state.F)
    )
    return kalman_state
end

function regression_update_kalman_state!(kalman_state, y, Z)
    if isnan(y)
        kalman_state.v = NaN
    else
        regression_update_v!(kalman_state, y, Z)
        update_llk!(kalman_state)
    end
    return kalman_state
end

function optim_kalman_filter(
    sys::LinearUnivariateTimeVariant{Fl}, filter::RegressionKalmanFilter{Fl}
) where Fl
    return filter_recursions!(filter.kalman_state, sys)
end

function kalman_filter!(
    filter_output::FilterOutput, sys::StateSpaceSystem, filter::RegressionKalmanFilter{Fl}
) where Fl
    filter_recursions!(filter_output, filter.kalman_state, sys)
    return filter_output
end

function filter_recursions!(
    kalman_state::RegressionKalmanState{Fl}, sys::LinearUnivariateTimeVariant{Fl}
) where Fl
    kalman_state.F = sys.H[1]
    @inbounds for t in eachindex(sys.y)
        regression_update_kalman_state!(kalman_state, sys.y[t], sys.Z[t])
    end
    return kalman_state.llk
end

function filter_recursions!(
    filter_output::FilterOutput,
    kalman_state::RegressionKalmanState{Fl},
    sys::LinearUnivariateTimeVariant,
) where Fl
    kalman_state.F = sys.H[1]
    @inbounds for t in eachindex(sys.y)
        regression_update_kalman_state!(kalman_state, sys.y[t], sys.Z[t])
    end
    save_kalman_state_in_filter_output!(filter_output, kalman_state)
    return filter_output
end
