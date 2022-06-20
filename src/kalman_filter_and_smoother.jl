# This file define interfaces with the filters defined in the filters folder
abstract type KalmanFilter end

const HALF_LOG_2_PI = 0.5 * log(2 * pi)

# Default loglikelihood function for optimization
function optim_loglike(
    model::StateSpaceModel, filter::KalmanFilter, unconstrained_hyperparameters::Vector{Fl}
) where Fl
    reset_filter!(filter)
    update_model_hyperparameters!(model, unconstrained_hyperparameters)
    update_filter_hyperparameters!(filter, model)
    # A way to stabilize the objective function is to take the mean of the
    # log like only for the optimizer
    return optim_kalman_filter(model.system, filter) / size(model.system.y, 1)
end

function update_model_hyperparameters!(
    model::StateSpaceModel, unconstrained_hyperparameters::Vector{Fl}
) where Fl
    register_unconstrained_values!(model, unconstrained_hyperparameters)
    constrain_hyperparameters!(model)
    fill_model_system!(model)
    return model
end

function update_filter_hyperparameters!(filter::KalmanFilter, model::StateSpaceModel)
    fill_model_filter!(filter, model)
    return filter
end

function fill_model_filter!(::KalmanFilter, ::StateSpaceModel)
    return nothing
end

function loglike(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
    return optim_loglike(model, filter, get_free_unconstrained_values(model)) * size(model.system.y, 1)
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
struct FilterOutput{Fl<:AbstractFloat}
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

function assert_possible_to_filter(model::StateSpaceModel)
    if !has_hyperparameter(model) || isfitted(model)
        return true
    end
    error("Model has not been estimated yet, please use `fit!`.")
end

get_standard_residuals(fo::FilterOutput) = get_standard_innovations(fo)
function get_standard_innovations(fo::FilterOutput)
    v = get_innovations(fo)
    F = get_innovations_variance(fo)
    @assert size(v, 2) == 1 # Must be univariate
    return v[:, 1] ./ sqrt.(F[:, 1, 1])
end

function vector_of_vector_to_matrix(vov::Vector{Vector{T}}) where T
    n = length(vov)
    p = length(vov[1])
    mat = Matrix{T}(undef, n, p)
    for i in 1:n, j in 1:p
        mat[i, j] = vov[i][j]
    end
    return mat
end

function vector_of_matrix_to_array(vom::Vector{Matrix{T}}) where T
    n = length(vom)
    p = length(vom[1])
    arr = Array{T, 3}(undef, n, p, p)
    for i in 1:n, j in 1:p, k in 1:p
        arr[i, j, k] = vom[i][j, k]
    end
    return arr
end

"""
    get_innovations

Returns the innovations `v` obtained with the Kalman filter.
"""
function get_innovations end

function get_innovations(filter::FilterOutput)
    return vector_of_vector_to_matrix(filter.v)
end

function get_innovations(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
    assert_possible_to_filter(model)
    return get_innovations(kalman_filter(model; filter=filter))
end

"""
    get_innovations_variance

Returns the variance `F` of innovations obtained with the Kalman filter.
"""
function get_innovations_variance end

function get_innovations_variance(filter::FilterOutput)
    return vector_of_matrix_to_array(filter.F)
end

function get_innovations_variance(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
    assert_possible_to_filter(model)
    return get_innovations_variance(kalman_filter(model; filter=filter))
end

"""
    get_filtered_state

Returns the filtered state `att` obtained with the Kalman filter.
"""
function get_filtered_state end

function get_filtered_state(filter::FilterOutput)
    return vector_of_vector_to_matrix(filter.att)
end

function get_filtered_state(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
    assert_possible_to_filter(model)
    return get_filtered_state(kalman_filter(model; filter=filter))
end

"""
    get_filtered_state_variance

Returns the variance `Ptt` of the filtered state obtained with the Kalman filter.
"""
function get_filtered_state_variance end

get_filtered_state_variance(filter::FilterOutput) = vector_of_matrix_to_array(filter.Ptt)

function get_filtered_state_variance(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
    assert_possible_to_filter(model)
    return get_filtered_state_variance(kalman_filter(model; filter=filter))
end

"""
    get_predictive_state

Returns the predictive state `a` obtained with the Kalman filter.
"""
function get_predictive_state end

get_predictive_state(filter::FilterOutput) = vector_of_vector_to_matrix(filter.a)

function get_predictive_state(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
    assert_possible_to_filter(model)
    return get_predictive_state(kalman_filter(model; filter=filter))
end

"""
    get_predictive_state_variance

Returns the variance `P` of the predictive state obtained with the Kalman filter.
"""
function get_predictive_state_variance end

get_predictive_state_variance(filter::FilterOutput) = vector_of_matrix_to_array(filter.P)

function get_predictive_state_variance(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
    assert_possible_to_filter(model)
    return get_predictive_state_variance(kalman_filter(model; filter=filter))
end

"""
    kalman_filter(model::StateSpaceModel; filter::KalmanFilter=default_filter(model)) -> FilterOutput

Filter the data using the desired `filter` and the estimated hyperparameters in `model`.
"""
function kalman_filter(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
    assert_possible_to_filter(model)
    filter_output = FilterOutput(model)
    reset_filter!(filter)
    if has_hyperparameter(model)
        free_unconstrained_values = get_free_unconstrained_values(model)
        update_model_hyperparameters!(model, free_unconstrained_values)
        update_filter_hyperparameters!(filter, model)
    end
    return kalman_filter!(filter_output, model.system, filter)
end

"""
    SmootherOutput{Fl<:Real}

Structure with the results of the smoother:
* `alpha`: smoothed state
* `V`: variance of smoothed state
"""
struct SmootherOutput{Fl<:Real}
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
    kalman_smoother(model::StateSpaceModel; filter::KalmanFilter=default_filter(model)) -> FilterOutput

Apply smoother to data filtered by `filter` and the estimated hyperparameters in `model`.
"""
function kalman_smoother(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
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

get_smoothed_state(smoother::SmootherOutput) = vector_of_vector_to_matrix(smoother.alpha)

function get_smoothed_state(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
    assert_possible_to_filter(model)
    return get_smoothed_state(kalman_smoother(model; filter=filter))
end

"""
    get_smoothed_state_variance

Returns the variance `V` of the smoothed state obtained with the smoother.
"""
function get_smoothed_state_variance end

get_smoothed_state_variance(smoother::SmootherOutput) = vector_of_matrix_to_array(smoother.V)

function get_smoothed_state_variance(model::StateSpaceModel; filter::KalmanFilter=default_filter(model))
    assert_possible_to_filter(model)
    return get_smoothed_state_variance(kalman_smoother(model; filter=filter))
end