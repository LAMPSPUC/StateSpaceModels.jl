mutable struct Forecast{Fl}
    expected_value::Vector{Vector{Fl}}
    covariance::Vector{Matrix{Fl}}
end

function forecast_expected_value(forec::Forecast)
    return permutedims(cat(forec.expected_value...; dims = 2))
end

"""
    forecast(model::SSM, steps_ahead::Int) where SSM

Forecast the mean and covariance for future observations from a StateSpaceModel (SSM).
"""
function forecast(model::SSM, steps_ahead::Int;
                  filter::KalmanFilter = default_filter(model)) where SSM
    # Query the type of model elements
    Fl = typeof_model_elements(model)
    # Observations to forecast
    forecasting_y = [model.system.y; fill(NaN, steps_ahead)]
    # Copy hyperparameters
    model_hyperparameters = deepcopy(model.hyperparameters)
    # Instantiate a new model
    forecasting_model = reinstantiate(model, forecasting_y)
    # Associate with the model hyperparameters
    forecasting_model.hyperparameters = model_hyperparameters
    # Perform the kalman filter
    fo = kalman_filter(forecasting_model)

    # fill forecast matrices
    expected_value = Vector{Vector{Fl}}(undef, steps_ahead)
    covariance = Vector{Matrix{Fl}}(undef, steps_ahead)
    for i in 1:steps_ahead
        expected_value[i] = [dot(model.system.Z, fo.a[end - steps_ahead + i]) + model.system.d]
        covariance[i] = fo.F[end - steps_ahead + i]
    end

    return Forecast{Fl}(expected_value, covariance)
end

"""
    simulate_scenarios(
        model::StateSpaceModel, n_periods::Int, n_scenarios::Int;
        filter::KalmanFilter=default_filter(model)
    ) -> Array{<:AbstractFloat, 3}

Samples `n_scenarios` future scenarios via Monte Carlo simulation for `n_periods` steps
ahead using the desired `filter`.
"""
function simulate_scenarios(
    model::StateSpaceModel, n_periods::Int, n_scenarios::Int;
    filter::KalmanFilter=default_filter(model)
)
    # Query the type of model elements
    Fl = typeof_model_elements(model)
    fo = kalman_filter(model)
    last_state = fo.a[end]
    num_series = size(model.system.y, 2)

    scenarios = Array{Fl, 3}(undef, n_periods, num_series, n_scenarios)
    for s in 1:n_scenarios
        scenarios[:, :, s] = simulate(model.system, last_state, n_periods)
    end
    return scenarios
end
