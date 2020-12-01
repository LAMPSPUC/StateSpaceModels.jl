function monotone_forecast_variance(forec::StateSpaceModels.Forecast)
    for i in 2:length(forec.covariance)
        if all(diag(forec.covariance[i]) .< diag(forec.covariance[i - 1]))
            return false
        end
    end
    return true
end

function expected_value_of_scenarios(scenarios::Array{T,3}) where T
    expected_values = Vector{Vector{T}}(undef, size(scenarios, 1))
    for t in 1:length(expected_values)
        expected_values[t] = vec(mean(scenarios[t, :, :]; dims=2))
    end
    return expected_values
end

function test_scenarios_adequacy_with_forecast(
    forec::StateSpaceModels.Forecast, scenarios::Array{T,3}; rtol::T=5e-3, atol::T=NaN
) where T
    # Test expected values
    if isnan(atol)
        for t in 1:length(forec.expected_value)
            @test forec.expected_value[t] ≈ expected_value_of_scenarios(scenarios)[t] rtol =
                rtol
        end
    else
        for t in 1:length(forec.expected_value)
            @test forec.expected_value[t] ≈ expected_value_of_scenarios(scenarios)[t] atol =
                atol
        end
    end
    return nothing
end
