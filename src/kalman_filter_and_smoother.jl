# This file define interfaces with the filters defined in the filters folder
abstract type KalmanFilter end

const HALF_LOG_2_PI = 0.5 * log(2 * pi)

# Default loglikelihood function for optimization
function optim_loglike(model::StateSpaceModel,
                       filter::KalmanFilter,
                       unconstrained_hyperparameters::Vector{Fl}) where Fl
    reset_filter!(filter)
    update_model_hyperparameters!(model, unconstrained_hyperparameters)
    update_filter_hyperparameters!(filter, model)
    return optim_kalman_filter(model.system, filter)
end

function update_model_hyperparameters!(model::StateSpaceModel,
                                       unconstrained_hyperparameters::Vector{Fl}) where Fl
    register_unconstrained_values!(model, unconstrained_hyperparameters)
    constrain_hyperparameters!(model)
    fill_model_system!(model)
    return nothing
end

function update_filter_hyperparameters!(filter::KalmanFilter, model::StateSpaceModel)
    fill_model_filter!(filter, model)
    return nothing
end
function fill_model_filter!(::KalmanFilter, ::StateSpaceModel)
    return nothing
end

function loglike(model::StateSpaceModel;
                 filter::KalmanFilter = default_filter(model))
    return optim_loglike(model, filter, get_free_unconstrained_values(model))
end

"""
    FilterOutput{Fl<:Real}

Structure with the results of the Kalman filter:
* `v`: innovations
* `F`: variance of innovations
* `a`: predictive state
* `att`: filtered state
* `P`: variance of predictive state
* `Ptt`: variance of filtered state
* `Pinf`: diffuse part of the covariance
"""
mutable struct FilterOutput{Fl<:Real}
    v::Vector{Vector{Fl}}
    F::Vector{Matrix{Fl}}
    a::Vector{Vector{Fl}}
    att::Vector{Vector{Fl}}
    P::Vector{Matrix{Fl}}
    Ptt::Vector{Matrix{Fl}}
    Pinf::Vector{Matrix{Fl}}

    function FilterOutput(model::StateSpaceModel)
        Fl = typeof_model_elements(model)
        n = size(model.system.y, 1)

        v = Vector{Vector{Fl}}(undef, n)
        F = Vector{Matrix{Fl}}(undef, n)
        a = Vector{Vector{Fl}}(undef, n + 1)
        att = Vector{Vector{Fl}}(undef, n)
        P = Vector{Matrix{Fl}}(undef, n + 1)
        Ptt = Vector{Matrix{Fl}}(undef, n)
        Pinf = Vector{Matrix{Fl}}(undef, n)
        return new{Fl}(v, F, a, att, P, Ptt, Pinf)
    end
end

"""
Returns the innovations `v` obtained with the Kalman filter.
"""
function get_innovations end

get_innovations(filter::FilterOutput) = permutedims(cat(filter.v...; dims=2))

function get_innovations(model::StateSpaceModel)
    isempty(model.results) && error("Model has not been estimated yet, please use `fit!`.")
    return get_innovations(kalman_filter(model))
end

"""
Returns the variance `F` of innovations obtained with the Kalman filter.
"""
function get_innovation_variance end

get_innovation_variance(filter_output::FilterOutput) = cat(filter_output.F...; dims=3)

function get_innovation_variance(model::StateSpaceModel)
    isempty(model.results) && error("Model has not been estimated yet, please use `fit!`.")
    return get_innovation_variance(kalman_filter(model))
end

"""
Returns the filtered state `att` obtained with the Kalman filter.
"""
function get_filtered_state end

get_filtered_state(filter::FilterOutput) = permutedims(cat(filter.att...; dims=2))

function get_filtered_state(model::StateSpaceModel)
    isempty(model.results) && error("Model has not been estimated yet, please use `fit!`.")
    return get_filtered_state(kalman_filter(model))
end

"""
Returns the filtered covariance `Ptt` of the states obtained with the Kalman filter.
"""
function get_filtered_covariance end

get_filtered_covariance(filter::FilterOutput) = cat(filter.Ptt...; dims=3)

function get_filtered_covariance(model::StateSpaceModel)
    isempty(model.results) && error("Model has not been estimated yet, please use `fit!`.")
    return get_filtered_covariance(kalman_filter(model))
end

"""
TODO
"""
one_step_ahead_predictions(filter_output::FilterOutput) = permutedims(cat(filter_output.a...; dims = 2))

"""
TODO
"""
covariance_one_step_ahead_predictions(filter_output::FilterOutput) = cat(filter_output.P...; dims = 3)

"""
TODO
"""
function kalman_filter(model::StateSpaceModel;
                       filter::KalmanFilter = default_filter(model))
    filter_output = FilterOutput(model)
    reset_filter!(filter)
    free_unconstrained_values = get_free_unconstrained_values(model)
    update_model_hyperparameters!(model, free_unconstrained_values)
    update_filter_hyperparameters!(filter, model)
    return kalman_filter!(filter_output, model.system, filter)
end

"""
TODO
"""
mutable struct SmootherOutput{Fl <: Real}
    alpha::Vector{Vector{Fl}}
    V::Vector{Matrix{Fl}}

    function SmootherOutput(model::StateSpaceModel)
        Fl = typeof_model_elements(model)
        n = size(model.system.y, 1)

        alpha = Vector{Vector{Fl}}(undef, n)
        V = Vector{Matrix{Fl}}(undef, n)
        return new{Fl}(alpha, V)
    end
end

"""
TODO
"""
function kalman_smoother(model::StateSpaceModel;
                         filter::KalmanFilter = default_filter(model))
    filter_output = FilterOutput(model)
    kalman_filter!(filter_output, model.system, filter)
    smoother_output = SmootherOutput(model)
    return kalman_smoother!(smoother_output, model.system, filter_output)
end

"""
TODO
"""
smoothed_estimates(smoother_output::SmootherOutput) = permutedims(cat(smoother_output.alpha...; dims = 2))

"""
TODO
"""
covariance_smoothed_estimates(smoother_output::SmootherOutput) = cat(smoother_output.V...; dims = 3)
