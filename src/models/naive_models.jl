abstract type NaiveModel end

function isfitted(model::NaiveModel)
    return model.fitted
end

isunivariate(model::NaiveModel) = true
length_observations(model::NaiveModel) = length(model.y)
observations(model::NaiveModel) = model.y
get_standard_residuals(model::NaiveModel) = model.residuals ./ sqrt(model.sigma2)
typeof_model_elements(model::NaiveModel) = eltype(model.y)

function assert_zero_missing_values(model::NaiveModel)
    for i in 1:length(model.y)
        if isnan(y[i])
            return error("model $(typeof(model)) does not support missing values.)")
        end
    end
    return nothing
end

@doc raw"""
    Naive(y::Vector{<:Real})

# TODO

# References
 * Hyndman, Rob J., Athanasopoulos, George. "Forecasting: Principles and Practice"
"""
mutable struct Naive <: NaiveModel
    y::Vector{<:Real}
    residuals::Vector{<:Real}
    sigma2::Real
    fitted::Bool

    function Naive(y::Vector{<:Real})
        Fl = eltype(y)
        n = length(y)
        return new(y, zeros(Fl, n), zero(Fl), false)
    end
end

function fit!(model::Naive)
    residuals = model.y[2:end] - model.y[1:end-1]
    model.residuals = residuals
    model.sigma2 = var(residuals)
    model.fitted = true
    return model
end

function forecast(model::Naive, steps_ahead::Int)
    @assert isfitted(model)
    Fl = eltype(model.y)
    expected_value = Vector{Vector{Fl}}(undef, steps_ahead)
    covariance = Vector{Matrix{Fl}}(undef, steps_ahead)
    for i in 1:steps_ahead
        expected_value[i] = [model.y[end]]
        covariance[i] = [model.sigma2 * i][:, :]
    end
    return Forecast{Fl}(expected_value, covariance)
end

function simulate_scenarios(model::Naive, steps_ahead::Int, n_scenarios::Int)
    Fl = typeof_model_elements(model)
    scenarios = Array{Fl, 3}(undef, steps_ahead, num_series, n_scenarios)
    for s in 1:n_scenarios, i in 1:steps_ahead
        if i == 1
            scenarios[i, 1, s] = model.y[end] + rand(model.residuals)
        else
            scenarios[i, 1, s] = scenarios[i-1, 1, s] + rand(model.residuals)
        end
    end
    return scenarios
end

@doc raw"""
    SeasonalNaive(y::Vector{<:Real}, seasoanl::Int)

# TODO

# References
 * Hyndman, Rob J., Athanasopoulos, George. "Forecasting: Principles and Practice"
"""
mutable struct SeasonalNaive <: NaiveModel
    y::Vector{<:Real}
    residuals::Vector{<:Real}
    seasonal::Int
    sigma2::Real
    fitted::Bool

    function SeasonalNaive(y::Vector{<:Real}, seasonal::Int)
        Fl = eltype(y)
        n = length(y)
        return new(y, zeros(Fl, n), seasonal, zero(Fl), false)
    end
end

function fit!(model::SeasonalNaive)
    residuals = model.y[seasonal+1:end] - model.y[1:end-seasonal]
    model.residuals = residuals
    model.sigma2 = var(residuals)
    model.fitted = true
    return model
end

function forecast(model::SeasonalNaive, steps_ahead::Int)
    @assert isfitted(model)
    Fl = eltype(model.y)
    expected_value = Vector{Vector{Fl}}(undef, steps_ahead)
    covariance = Vector{Matrix{Fl}}(undef, steps_ahead)
    for i in 1:steps_ahead
        expected_value[i] = [model.y[end - seasonal + i]]
        covariance[i] = [model.sigma2 * (floor(Int, (i - 1)/model.seasonal) + 1)][:, :]
    end
    return Forecast{Fl}(expected_value, covariance)
end

function simulate_scenarios(model::SeasonalNaive, steps_ahead::Int, n_scenarios::Int)
    Fl = typeof_model_elements(model)
    scenarios = Array{Fl, 3}(undef, steps_ahead, num_series, n_scenarios)
    #TODO
    return scenarios
end

function backtest(model::NaiveModel, steps_ahead::Int, start_idx::Int;
                  n_scenarios::Int = 10_000)
    Fl = typeof_model_elements(model)
    num_fits = length(model.system.y) - start_idx - steps_ahead
    b = Backtest{Fl}(num_mle, steps_ahead)
    for i in 1:num_mle
        println("Backtest: step $i of $num_mle")
        y_to_fit = model.system.y[1:start_idx - 1 + i]
        y_to_verify = model.system.y[start_idx + i:start_idx - 1 + i + steps_ahead]
        model_to_fit = reinstantiate(model, y_to_fit)
        fit!(model_to_fit)
        forec = forecast(model_to_fit, steps_ahead; filter=filter)
        scenarios = simulate_scenarios(model_to_fit, steps_ahead, n_scenarios)
        expected_value_vector = forecast_expected_value(forec)[:]
        abs_errors = evaluate_abs_error(y_to_verify, expected_value_vector)
        crps_scores = evaluate_crps(y_to_verify, scenarios[:, 1, :])
        b.abs_errors[:, i] = abs_errors
        b.crps_scores[:, i] = crps_scores
    end
    for i in 1:steps_ahead
        b.mae[i] = mean(b.abs_errors[i, :])
        b.mean_crps[i] = mean(b.crps_scores[i, :])
    end
    return b
end