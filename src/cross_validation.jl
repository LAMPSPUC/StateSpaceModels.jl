struct CrossValidation{Fl <: AbstractFloat}
    abs_errors::Matrix{Fl}
    mae::Vector{Fl}
    crps_scores::Matrix{Fl}
    mean_crps::Vector{Fl}
    function CrossValidation{Fl}(n::Int, steps_ahead::Int) where Fl
        abs_errors = Matrix{Fl}(undef, steps_ahead, n)
        crps_scores = Matrix{Fl}(undef, steps_ahead, n)
        mae = Vector{Fl}(undef, steps_ahead)
        mean_crps = Vector{Fl}(undef, steps_ahead)
        return new(abs_errors, mae, crps_scores, mean_crps)
    end
end

discrete_crps_indicator_function(val::Fl, z::Fl) where {Fl} = val < z
function crps(val::Fl, scenarios::Vector{Fl}) where {Fl}
    sorted_scenarios = sort(scenarios)
    m = length(scenarios)
    crps_score = zero(Fl)
    for i = 1:m
        crps_score +=
            (sorted_scenarios[i] - val) *
            (m * discrete_crps_indicator_function(val, sorted_scenarios[i]) - i + 0.5)
    end
    return (2 / m^2) * crps_score
end
evaluate_abs_error(y::Vector{Fl}, forecast::Vector{Fl}) where Fl = abs.(y - forecast)
function evaluate_crps(y::Vector{Fl}, scenarios::Matrix{Fl}) where {Fl}
    crps_scores = Vector{Fl}(undef, length(y))
    for k = 1:length(y)
        crps_scores[k] = crps(y[k], scenarios[k, :])
    end
    return crps_scores
end

"""
    cross_validation(model::StateSpaceModel, steps_ahead::Int, start_idx::Int;
             n_scenarios::Int = 10_000,
             filter::KalmanFilter=default_filter(model),
             optimizer::Optimizer=default_optimizer(model)) where Fl

Makes rolling window estimating and forecasting to benchmark the forecasting skill of the model
in for different time periods and different lead times. The function returns a struct with the MAE
and mean CRPS per lead time. See more on [Cross validation of the forecasts of a model](@ref)

# References
 * DTU course "31761 - Renewables in electricity markets" available on youtube https://www.youtube.com/watch?v=Ffo8XilZAZw&t=556s
"""
function cross_validation(model::StateSpaceModel, steps_ahead::Int, start_idx::Int;
                  n_scenarios::Int = 10_000,
                  filter::KalmanFilter=default_filter(model),
                  optimizer::Optimizer=default_optimizer(model))
    Fl = typeof_model_elements(model)
    num_mle = length(model.system.y) - start_idx - steps_ahead
    cv = CrossValidation{Fl}(num_mle, steps_ahead)
    for i in 1:num_mle
        println("CrossValidation: step $i of $num_mle")
        y_to_fit = model.system.y[1:start_idx - 1 + i]
        y_to_verify = model.system.y[start_idx + i:start_idx - 1 + i + steps_ahead]
        model_to_fit = reinstantiate(model, y_to_fit)
        fit!(model_to_fit; filter=filter, optimizer=optimizer, save_hyperparameter_distribution=false)
        forec = forecast(model_to_fit, steps_ahead; filter=filter)
        scenarios = simulate_scenarios(model_to_fit, steps_ahead, n_scenarios; filter=filter)
        expected_value_vector = forecast_expected_value(forec)[:]
        abs_errors = evaluate_abs_error(y_to_verify, expected_value_vector)
        crps_scores = evaluate_crps(y_to_verify, scenarios[:, 1, :])
        cv.abs_errors[:, i] = abs_errors
        cv.crps_scores[:, i] = crps_scores
    end
    for i in 1:steps_ahead
        cv.mae[i] = mean(cv.abs_errors[i, :])
        cv.mean_crps[i] = mean(cv.crps_scores[i, :])
    end
    return cv
end