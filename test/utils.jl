read_csv(file::String) = DataFrame!(CSV.File(file))

function monotone_forecast_variance(forec::StateSpaceModels.Forecast)
    for i in 2:length(forec.covariance)
        if all(diag(forec.covariance[i]) .<= diag(forec.covariance[i - 1]))
            return false
        end
    end
    return true
end

function expected_value_of_scenarios(scenarios::Array{T,3}) where T
    expected_values = Vector{Vector{T}}(undef, size(scenarios, 1))
    for t in 1:length(expected_values)
        expected_values[t] = vec(mean(scenarios[t, :, :], dims=2))
    end
    return expected_values
end

function test_scenarios_adequacy_with_forecast(forec::StateSpaceModels.Forecast, 
                                               scenarios::Array{T,3}; 
                                               expected_value_rtol::T=1e-2,
                                               quantiles_rtol::T=5e-2) where T
    # Test expected values
    for t in 1:length(forec.expected_value)
        @test forec.expected_value[t] ≈ expected_value_of_scenarios(scenarios)[t] rtol = expected_value_rtol
    end
    # Test quantiles
    # Default quantiles to test
    # quantiles = collect(0.05:0.05:0.95)
    # for q in quantiles, t in 1:length(forec.expected_value)
    #     # Distributions parametrize Normal with μ and σ
    #     dist = Normal(forec.expected_value[t][1], sqrt(forec.covariance[t][1]))
    #     @test quantile(dist, q) ≈ quantile(scenarios[t, 1, :], q) rtol = quantiles_rtol
    # end
    return
end