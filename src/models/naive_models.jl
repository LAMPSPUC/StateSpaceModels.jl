abstract type NaiveModel end

function isfitted(model::NaiveModel)
    return model.fitted
end

isunivariate(model::NaiveModel) = true
num_observations(model::NaiveModel) = length(model.y)
observations(model::NaiveModel) = model.y
get_standard_residuals(model::NaiveModel) = model.residuals ./ sqrt(model.sigma2)
typeof_model_elements(model::NaiveModel) = eltype(model.y)

function assert_zero_missing_values(model::NaiveModel)
    for i in 1:length(model.y)
        if isnan(model.y[i])
            return error("model $(typeof(model)) does not support missing values.")
        end
    end
    return nothing
end

@doc raw"""
    Naive(y::Vector{<:Real})

A naive model where the h step ahead forecast is 
```math
y_{T+h|T} = y_T
```

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
    assert_zero_missing_values(model)
    residuals = model.y[2:end] - model.y[1:end-1]
    model.residuals = residuals
    model.sigma2 = var(residuals)
    model.fitted = true
    return model
end

function forecast(model::Naive, steps_ahead::Int; bootstrap::Bool = false)
    @assert isfitted(model)
    Fl = eltype(model.y)
    expected_value = Vector{Vector{Fl}}(undef, steps_ahead)
    covariance = Vector{Matrix{Fl}}(undef, steps_ahead)
    if bootstrap
        scenarios = simulate_scenarios(model, steps_ahead, 10_000)
        for i in 1:steps_ahead
            expected_value[i] = [mean(scenarios[i, 1, :])]
            covariance[i] = [var(scenarios[i, 1, :])][:, :]
        end
    else
        for i in 1:steps_ahead
            expected_value[i] = [model.y[end]]
            covariance[i] = [model.sigma2 * i][:, :]
        end
    end
    return Forecast{Fl}(expected_value, covariance)
end

function simulate_scenarios(model::Naive, steps_ahead::Int, n_scenarios::Int)
    Fl = typeof_model_elements(model)
    scenarios = Array{Fl, 3}(undef, steps_ahead, 1, n_scenarios)
    for s in 1:n_scenarios, i in 1:steps_ahead
        scenarios[i, 1, s] = model.y[end] + rand(model.residuals)
    end
    return scenarios
end

function reinstantiate(::Naive, y::Vector{<:Real})
    return Naive(y)
end

@doc raw"""
    SeasonalNaive(y::Vector{<:Real}, seasoanl::Int)

A seasonal naive model where the h step ahead forecast is 
```math
y_{T+h|T} = y_{T + h - m(k+1)}
```
where m is the seasonal period and k is the integer part of (h-1)/m.

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
    assert_zero_missing_values(model)
    residuals = model.y[model.seasonal+1:end] - model.y[1:end-model.seasonal]
    model.residuals = residuals
    model.sigma2 = var(residuals)
    model.fitted = true
    return model
end

function forecast(model::SeasonalNaive, steps_ahead::Int; bootstrap::Bool = false)
    @assert isfitted(model)
    Fl = eltype(model.y)
    expected_value = Vector{Vector{Fl}}(undef, steps_ahead)
    covariance = Vector{Matrix{Fl}}(undef, steps_ahead)
    if bootstrap
        scenarios = simulate_scenarios(model, steps_ahead, 10_000)
        for i in 1:steps_ahead
            expected_value[i] = [mean(scenarios[i, 1, :])]
            covariance[i] = [var(scenarios[i, 1, :])][:, :]
        end
    else
        for i in 1:steps_ahead
            if i in 1:model.seasonal
                expected_value[i] = [model.y[end - model.seasonal + i]]
            else
                expected_value[i] = copy(expected_value[i - model.seasonal])
            end
            covariance[i] = [model.sigma2 * (fld(i - 1, model.seasonal) + 1)][:, :]
        end
    end
    return Forecast{Fl}(expected_value, covariance)
end

function simulate_scenarios(model::SeasonalNaive, steps_ahead::Int, n_scenarios::Int)
    Fl = typeof_model_elements(model)
    scenarios = Array{Fl, 3}(undef, steps_ahead, 1, n_scenarios)
    for s in 1:n_scenarios, i in 1:steps_ahead
        scenarios[i, 1, s] = model.y[end - model.seasonal + mod(i, model.seasonal)] + rand(model.residuals)
    end
    return scenarios
end

function reinstantiate(model::SeasonalNaive, y::Vector{<:Real})
    return SeasonalNaive(y, model.seasonal)
end

@doc raw"""
    ExperimentalSeasonalNaive(y::Vector{<:Real}, seasonal::Int; S::Int = 10_000)

A seasonal naive model where the h step ahead forecast is the mean of the simulation 
of S scenarios 
```math
y_{T+h|T} = y_{T + h - m(k+1)} + \varepsilon_t
```
where m is the seasonal period, k is the integer part of (h-1)/m and ``\varepsilon_t`` is
a sampled error. 

We call it experimental because so far we could not find a good reference and implementation.
If you know something please post it as an issue.
"""
mutable struct ExperimentalSeasonalNaive <: NaiveModel
    y::Vector{<:Real}
    residuals::Vector{<:Real}
    seasonal::Int
    sigma2::Real
    fitted::Bool
    S::Int

    function ExperimentalSeasonalNaive(y::Vector{<:Real}, seasonal::Int; S::Int = 10_000)
        Fl = eltype(y)
        n = length(y)
        return new(y, zeros(Fl, n), seasonal, zero(Fl), false, S)
    end
end

function fit!(model::ExperimentalSeasonalNaive)
    assert_zero_missing_values(model)
    residuals = model.y[model.seasonal+1:end] - model.y[1:end-model.seasonal]
    model.residuals = residuals
    model.sigma2 = var(residuals)
    model.fitted = true
    return model
end

function forecast(model::ExperimentalSeasonalNaive, steps_ahead::Int)
    @assert isfitted(model)
    Fl = eltype(model.y)
    expected_value = Vector{Vector{Fl}}(undef, steps_ahead)
    covariance = Vector{Matrix{Fl}}(undef, steps_ahead)
    scenarios = simulate_scenarios(model, steps_ahead, model.S)
    for i in 1:steps_ahead
        expected_value[i] = [mean(scenarios[i, 1, :])]
        covariance[i] = [var(scenarios[i, 1, :])][:, :]
    end
    return Forecast{Fl}(expected_value, covariance)
end

function simulate_scenarios(model::ExperimentalSeasonalNaive, steps_ahead::Int, n_scenarios::Int)
    Fl = typeof_model_elements(model)
    scenarios = Array{Fl, 3}(undef, steps_ahead, 1, n_scenarios)
    for i in 1:steps_ahead
        if i <= model.seasonal
            for s in 1:n_scenarios
                scenarios[i, 1, s] = model.y[end - model.seasonal + i] + rand(model.residuals)
            end
        else
            for s in 1:n_scenarios
                scenarios[i, 1, s] = scenarios[i - model.seasonal, 1, s] + rand(model.residuals)
            end
        end
    end
    return scenarios
end

function reinstantiate(model::ExperimentalSeasonalNaive, y::Vector{<:Real})
    return ExperimentalSeasonalNaive(y, model.seasonal; S = model.S)
end

function cross_validation(model::NaiveModel, steps_ahead::Int, start_idx::Int;
                  n_scenarios::Int = 10_000)
    Fl = typeof_model_elements(model)
    num_fits = length(model.y) - start_idx - steps_ahead
    b = CrossValidation{Fl}(num_fits, steps_ahead)
    for i in 1:num_fits
        println("CrossValidation: step $i of $num_fits")
        y_to_fit = model.y[1:start_idx - 1 + i]
        y_to_verify = model.y[start_idx + i:start_idx - 1 + i + steps_ahead]
        model_to_fit = reinstantiate(model, y_to_fit)
        fit!(model_to_fit)
        forec = forecast(model_to_fit, steps_ahead)
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