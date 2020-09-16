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
    get_innovations

Returns the innovations `v` obtained with the Kalman filter.
"""
function get_innovations end

get_innovations(filter::FilterOutput) = permutedims(cat(filter.v...; dims=2))

function get_innovations(model::StateSpaceModel; filter=default_filter(model))
    isfitted(model) || error("Model has not been estimated yet, please use `fit!`.")
    return get_innovations(kalman_filter(model; filter=filter))
end

"""
    get_innovation_variance

Returns the variance `F` of innovations obtained with the Kalman filter.
"""
function get_innovation_variance end

get_innovation_variance(filter_output::FilterOutput) = cat(filter_output.F...; dims=3)

function get_innovation_variance(model::StateSpaceModel; filter=default_filter(model))
    isfitted(model) || error("Model has not been estimated yet, please use `fit!`.")
    return get_innovation_variance(kalman_filter(model; filter=filter))
end

"""
    get_filtered_state

Returns the filtered state `att` obtained with the Kalman filter.
"""
function get_filtered_state end

get_filtered_state(filter::FilterOutput) = permutedims(cat(filter.att...; dims=2))

function get_filtered_state(model::StateSpaceModel; filter=default_filter(model))
    isfitted(model) || error("Model has not been estimated yet, please use `fit!`.")
    return get_filtered_state(kalman_filter(model; filter=filter))
end

"""
    get_filtered_variance

Returns the variance `Ptt` of the filtered state obtained with the Kalman filter.
"""
function get_filtered_variance end

get_filtered_variance(filter::FilterOutput) = cat(filter.Ptt...; dims=3)

function get_filtered_variance(model::StateSpaceModel; filter=default_filter(model))
    isfitted(model) || error("Model has not been estimated yet, please use `fit!`.")
    return get_filtered_variance(kalman_filter(model; filter=filter))
end

"""
    get_predictive_state

Returns the predictive state `a` obtained with the Kalman filter.
"""
function get_predictive_state end

get_predictive_state(filter::FilterOutput) = permutedims(cat(filter.a...; dims=2))

function get_predictive_state(model::StateSpaceModel; filter=default_filter(model))
    isfitted(model) || error("Model has not been estimated yet, please use `fit!`.")
    return get_predictive_state(kalman_filter(model); filter=filter)
end

"""
    get_predictive_variance

Returns the variance `P` of the predictive state obtained with the Kalman filter.
"""
function get_predictive_variance end

get_predictive_variance(filter::FilterOutput) = cat(filter.P...; dims=3)

function get_predictive_variance(model::StateSpaceModel; filter=default_filter(model))
    isfitted(model) || error("Model has not been estimated yet, please use `fit!`.")
    return get_predictive_variance(kalman_filter(model; filter=filter))
end

"""
    kalman_filter(model::StateSpaceModel; filter=default_filter(model)) -> FilterOutput

Filter the data using the desired `filter` and the estimated hyperparameters in `model`.
"""
function kalman_filter(model::StateSpaceModel; filter=default_filter(model))
    isfitted(model) || error("Model has not been estimated yet, please use `fit!`.")
    filter_output = FilterOutput(model)
    reset_filter!(filter)
    free_unconstrained_values = get_free_unconstrained_values(model)
    update_model_hyperparameters!(model, free_unconstrained_values)
    update_filter_hyperparameters!(filter, model)
    return kalman_filter!(filter_output, model.system, filter)
end

"""
    SmootherOutput{Fl<:Real}

Structure with the results of the smoother:
* `alpha`: smoothed state
* `V`: variance of smoothed state
"""
mutable struct SmootherOutput{Fl<:Real}
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
    kalman_smoother(model::StateSpaceModel; filter=default_filter(model)) -> FilterOutput

Apply smoother to data filtered by `filter` and the estimated hyperparameters in `model`.
"""
function kalman_smoother(model::StateSpaceModel; filter=default_filter(model))
    filter_output = FilterOutput(model)
    kalman_filter!(filter_output, model.system, filter)
    smoother_output = SmootherOutput(model)
    return kalman_smoother!(smoother_output, model.system, filter_output)
end

"""
    get_smoothed_state

Returns the smoothed state `alpha` obtained with the smoother.
"""
function get_smoothed_state end

get_smoothed_state(smoother::SmootherOutput) = permutedims(cat(smoother.alpha...; dims = 2))

function get_smoothed_state(model::StateSpaceModel; filter=default_filter(model))
    isfitted(model) || error("Model has not been estimated yet, please use `fit!`.")
    return get_smoothed_state(kalman_smoother(model; filter=filter))
end

"""
    get_smoothed_variance

Returns the variance `V` of the smoothed state obtained with the smoother.
"""
function get_smoothed_variance end

get_smoothed_variance(smoother::SmootherOutput) = cat(smoother.V...; dims = 3)

function get_smoothed_variance(model::StateSpaceModel; filter=default_filter(model))
    isfitted(model) || error("Model has not been estimated yet, please use `fit!`.")
    return get_smoothed_variance(kalman_smoother(model; filter=filter))
end
