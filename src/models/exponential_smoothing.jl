@doc raw"""
    ExponentialSmoothing(
        y::Vector{Fl}; 
        trend::Bool = false,
        damped_trend::Bool = false,
        seasonal::Int = 0
    ) where Fl

Linear exponential smoothing models. These models are also known as ETS in the literature.
This model is estimated using the Kalman filter for linear Gaussian state space models, for this
reason the possible models are the following ETS with additive errors:
 * ETS(A, N, N)
 * ETS(A, A, N)
 * ETS(A, Ad, N)
 * ETS(A, N, A)
 * ETS(A, A, A)
 * ETS(A, Ad, A)

Other softwares have use the augmented least squares approach and have all the possible ETS 
combinations. The Kalman filter approach might be slower than others but have the advantages of
filtering the components.

# References
 * Hyndman, Rob, Anne B. Koehler, J. Keith Ord, and Ralph D. Snyder.
   Forecasting with exponential smoothing: the state space approach.
   Springer Science & Business Media, 2008.
 * Hyndman, Robin John; Athanasopoulos, George. 
   Forecasting: Principles and Practice. 
   2nd ed. OTexts, 2018.
"""
mutable struct ExponentialSmoothing <: StateSpaceModel
    hyperparameters::HyperParameters
    trend::Bool
    damped_trend::Bool
    seasonal::Int
    system::LinearUnivariateTimeInvariant
    results::Results

    function ExponentialSmoothing(y::Vector{Fl}; 
                                  trend::Bool = false,
                                  damped_trend::Bool = false,
                                  seasonal::Int = 0
                                  ) where Fl
        @assert seasonal != 1 "seasonal must be different than 1"
        if damped_trend
            @assert trend
        end
        # Define system matrices
        Z = build_Z(Fl, trend, seasonal)
        T = build_T(Fl, trend, seasonal)
        R = zeros(Fl, size(T, 1), 1)
        R[1] = 1
        d = zero(Fl)
        c = zeros(Fl, size(T, 1))
        H = zero(Fl)
        Q = ones(Fl, 1, 1)

        system = LinearUnivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        # Define hyperparameters names
        names = build_names(trend, damped_trend, seasonal)
        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, trend, damped_trend, 
                   seasonal, system, Results{Fl}())
    end
end

num_states(trend::Bool, seasonal::Int) = 2 + trend + seasonal

function build_Z(Fl::DataType, trend::Bool, seasonal::Int)
    n_states = num_states(trend, seasonal)
    Z = zeros(Fl, n_states)
    Z[1:2] .= 1
    if seasonal > 0
        Z[3 + trend] = 1
    end
    return Z
end

function build_T(Fl::DataType, trend::Bool, seasonal::Int)
    n_states = num_states(trend, seasonal)
    T = zeros(Fl, n_states, n_states)
    T[2, 2] = 1
    if trend
        T[2:3, 3] .= 1
    end
    if seasonal > 0
        T[3 + trend, end] = 1
        for i in 3+trend+1:n_states
            T[i, i-1] = 1
        end
    end
    return T
end

function build_names(trend::Bool, damped_trend::Bool, seasonal::Int)
    names = String["sigma2", "smoothing_level"]
    if trend
        push!(names, "smoothing_trend")
        if damped_trend
            push!(names, "damping_trend")
        end
    end
    if seasonal > 0
        push!(names, "smoothing_seasonal")
    end
    push!(names, "initial_level")
    if trend
        push!(names, "initial_trend")
    end
    if seasonal > 0
        for i in 1:seasonal - 1
            push!(names, "initial_seasonal_$i")
        end
    end
    return names
end

function diff_es(y::Vector{Fl}, s::Int) where Fl
    # Seasonal differencing
    y = y[s:end] - y[1:end-s+1]
    return y
end

function default_filter(model::ExponentialSmoothing)
    Fl = typeof_model_elements(model)
    n_states = num_states(model.trend, model.seasonal)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, n_states)
    P1 = Fl(1e6) .* Matrix{Fl}(I, n_states, n_states)
    skip_llk = 0
    return UnivariateKalmanFilter(a1, P1, skip_llk, steadystate_tol)
end

function initial_hyperparameters!(model::ExponentialSmoothing)
    Fl = typeof_model_elements(model)
    observations = filter(!isnan, model.system.y)
    observed_variance = variance_of_valid_observations(model.system.y)
    initial_hyperparameters = Dict{String,Fl}(
        "sigma2" => observed_variance,
        "smoothing_level" => Fl(0.1),
        "initial_level" => observations[1]
    )
    if model.trend
        initial_hyperparameters["smoothing_trend"] = Fl(0.01)
        initial_hyperparameters["initial_trend"] = observations[2] - observations[1]
        if model.damped_trend
            initial_hyperparameters["damping_trend"] = Fl(0.97)
        end
    end
    if model.seasonal > 0
        # If the model is seasonal we update the initial_level and initial_trend
        initial_hyperparameters["smoothing_seasonal"] = Fl(0.01)
        # TODO this might be wrong for missing observations in the first s observations
        initial_hyperparameters["initial_level"] = mean(observations[1:model.seasonal - 1])
        if model.trend
            obs = diff_es(observations, model.seasonal)
            initial_hyperparameters["initial_trend"] = obs[2] - obs[1]
        end
        for i in 1:model.seasonal - 1
            initial_hyperparameters["initial_seasonal_$i"] = observations[i] - 
                                                             mean(observations[1:model.seasonal - 1])
        end
    end
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::ExponentialSmoothing)
    Fl = typeof_model_elements(model)
    constrain_variance!(model, "sigma2")
    constrain_box!(model, "smoothing_level", Fl(0.0001), Fl(0.9999))
    constrain_identity!(model, "initial_level")
    if model.trend
        ub = min(Fl(0.9999), get_constrained_value(model, "smoothing_level"))
        constrain_box!(model, "smoothing_trend", Fl(0.0001), ub)
        constrain_identity!(model, "initial_trend")
        if model.damped_trend
            constrain_box!(model, "damping_trend", Fl(0.8), Fl(0.98))
        end
    end
    if model.seasonal > 0
        ub = min(Fl(0.9999), 1 - min(Fl(0.99989), get_constrained_value(model, "smoothing_level")))
        constrain_box!(model, "smoothing_seasonal", Fl(0.0001), ub)
        for i in 1:model.seasonal - 1
            constrain_identity!(model, "initial_seasonal_$i")
        end
    end
    return model
end

function unconstrain_hyperparameters!(model::ExponentialSmoothing)
    Fl = typeof_model_elements(model)
    unconstrain_variance!(model, "sigma2")
    unconstrain_box!(model, "smoothing_level", Fl(0.0001), Fl(0.9999))
    unconstrain_identity!(model, "initial_level")
    if model.trend
        ub = min(Fl(0.9999), get_constrained_value(model, "smoothing_level"))
        unconstrain_box!(model, "smoothing_trend", Fl(0.0001), ub)
        unconstrain_identity!(model, "initial_trend")
        if model.damped_trend
            unconstrain_box!(model, "damping_trend", Fl(0.8), Fl(0.98))
        end
    end
    if model.seasonal > 0
        ub = min(Fl(0.9999), 1 - min(Fl(0.99989), get_constrained_value(model, "smoothing_level")))
        unconstrain_box!(model, "smoothing_seasonal", Fl(0.0001), ub)
        for i in 1:model.seasonal - 1
            unconstrain_identity!(model, "initial_seasonal_$i")
        end
    end
    return model
end

function fill_model_system!(model::ExponentialSmoothing)
    model.system.Q[1] = get_constrained_value(model, "sigma2")
    i = 1
    model.system.R[i] = 1 - get_constrained_value(model, "smoothing_level")
    i += 1
    model.system.R[i] = get_constrained_value(model, "smoothing_level")
    i += 1
    if model.trend
        model.system.R[i] = get_constrained_value(model, "smoothing_trend")
        i += 1
        if model.damped_trend
            model.system.T[2:3, 3] .= get_constrained_value(model, "damping_trend")
        end
    end
    if model.seasonal > 0
        model.system.R[1] -= get_constrained_value(model, "smoothing_seasonal")
        model.system.R[i] = get_constrained_value(model, "smoothing_seasonal")
    end
    return model
end

function fill_model_filter!(filter::KalmanFilter, model::ExponentialSmoothing)
    Fl = typeof_model_elements(model)
    initial_level = get_constrained_value(model, "initial_level")
    initial_state = [0; initial_level]
    if model.trend 
        initial_trend = get_constrained_value(model, "initial_trend")
        initial_state = vcat(initial_state, initial_trend)
    end
    if model.seasonal > 0
        initial_seasonal = Fl[]
        for i in 1:model.seasonal - 1
            push!(initial_seasonal, get_constrained_value(model, "initial_seasonal_$i"))
        end
        last_initial_seasonal = -sum(initial_seasonal)
        push!(initial_seasonal, last_initial_seasonal)
        initial_state = vcat(initial_state, initial_seasonal)
    end
    filter.kalman_state.a = model.system.T * initial_state
    filter.kalman_state.P = model.system.R * model.system.Q * model.system.R'
    return nothing
end

has_exogenous(model::ExponentialSmoothing) = false

function reinstantiate(model::ExponentialSmoothing, y::Vector{Fl}) where Fl
    return ExponentialSmoothing(y; 
                trend = model.trend,
                damped_trend = model.damped_trend,
                seasonal = model.seasonal
            )
end

function model_name(model::ExponentialSmoothing)
    E = "A"
    T = model.trend ? 
        model.damped_trend ? "Ad" : "A" :
        "N"
    S = model.seasonal > 0 ? "A" : "N"
    return "ETS($E,$T,$S)"
end

function dict_components(model::ExponentialSmoothing)
    dict_components = OrderedDict{String, Int}()
    i = 2
    dict_components["Trend"] = i
    i += 1
    if model.trend
        dict_components["Slope"] = i
        i += 1
    end
    if model.seasonal > 0
        dict_components["Seasonal"] = i
    end
    return dict_components
end

@doc raw"""
    auto_ets(y::Vector{Fl}; seasonal::Int = 0) where Fl

Automatically fits the best [`ExponentialSmoothing`](@ref) model according to the best AIC 
between the models:
 * ETS(A, N, N)
 * ETS(A, A, N)
 * ETS(A, Ad, N)

If the user provides the time series seasonality it will search between the models
 * ETS(A, N, A)
 * ETS(A, A, A)
 * ETS(A, Ad, A)

# References
 * Hyndman, Robin John; Athanasopoulos, George. 
 Forecasting: Principles and Practice. 
 2nd ed. OTexts, 2018.
"""
function auto_ets(y::Vector{Fl}; seasonal::Int = 0) where Fl
    models = StateSpaceModel[]
    models_aic = Fl[]
    @assert seasonal != 1 "seasonal must be different than 1"
    m1 = ExponentialSmoothing(y; trend = false, damped_trend = false, seasonal = seasonal)
    fit!(m1)
    push!(models, m1)
    push!(models_aic, m1.results.aic)
    m2 = ExponentialSmoothing(y; trend = true, damped_trend = false, seasonal = seasonal)
    fit!(m2)
    push!(models, m2)
    push!(models_aic, m2.results.aic)
    m3 = ExponentialSmoothing(y; trend = true, damped_trend = true, seasonal = seasonal)
    fit!(m3)
    push!(models, m3)
    push!(models_aic, m3.results.aic)
    best_aic_idx = findmin(models_aic)[2] # index of the best bic
    return models[best_aic_idx]
end