@testset "ARIMA" begin
    internet = read_csv(StateSpaceModels.INTERNET)
    dinternet = internet.dinternet[2:end]
    @test has_fit_methods(ARIMA)

    model = ARIMA(dinternet, (1, 0, 1))
    StateSpaceModels.fit(model)
    @test loglike(model) ≈ -254.149 atol = 1e-5 rtol = 1e-5

    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simualting
    # scenarios = simulate_scenarios(model, 10, 10_000)
    # test_scenarios_adequacy_with_forecast(forec, scenarios)

    missing_obs = [6, 16, 26, 36, 46, 56, 66, 72, 73, 74, 75, 76, 86, 96]
    missing_dinternet = copy(dinternet)
    missing_dinternet[missing_obs] .= NaN

    model = ARIMA(missing_dinternet, (1, 0, 1))
    StateSpaceModels.fit(model)
    @test_broken loglike(model) ≈ -225.770 atol = 1e-5 rtol = 1e-5

    wpi = read_csv(StateSpaceModels.WHOLESALE_PRICE_INDEX).wpi
    model = ARIMA(wpi, (1, 1, 1))
    StateSpaceModels.fit(model)
    @test loglike(model) ≈ -137.246818 atol = 1e-5 rtol = 1e-5
end