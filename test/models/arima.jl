@testset "ARIMA" begin
    internet = CSV.read(StateSpaceModels.INTERNET, DataFrame)
    dinternet = internet.dinternet[2:end]
    @test has_fit_methods(ARIMA)

    model = ARIMA(dinternet, (1, 0, 1))
    fit!(model)
    @test loglike(model) ≈ -254.149 atol = 1e-5 rtol = 1e-5

    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # Prediction from Pyhton statsmodels
    predicted_mean = [
        -0.72809028,
        -0.47352952,
        -0.30797034,
        -0.20029528,
        -0.13026644,
        -0.08472164,
        -0.05510058,
        -0.03583588,
        -0.02330665,
        -0.01515799,
    ]
    @test predicted_mean ≈ vcat(forec.expected_value...) atol = 1e-3
    # simualting
    scenarios = simulate_scenarios(model, 10, 100_000)
    # Values are very close to 0.0 so we test with absolute tolerance
    # It attains 1e-3 when we make 10M simulations, which is too much
    # computation for a rather simple test.
    test_scenarios_adequacy_with_forecast(forec, scenarios; atol=1e-1)

    missing_obs = [6, 16, 26, 36, 46, 56, 66, 72, 73, 74, 75, 76, 86, 96]
    missing_dinternet = copy(dinternet)
    missing_dinternet[missing_obs] .= NaN

    model = ARIMA(missing_dinternet, (1, 0, 1))
    fit!(model)
    @test loglike(model) ≈ -225.770 atol = 1e-5 rtol = 1e-5

    wpi = CSV.read(StateSpaceModels.WHOLESALE_PRICE_INDEX, DataFrame).wpi
    model = ARIMA(wpi, (1, 1, 1))
    fit!(model)
    @test loglike(model) ≈ -137.246818 atol = 1e-5 rtol = 1e-5
end
