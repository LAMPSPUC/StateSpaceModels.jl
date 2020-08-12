@testset "Basic Structural Model" begin
    air_passengers = read_csv(StateSpaceModels.AIR_PASSENGERS)
    log_air_passengers = log.(air_passengers.passengers)

    @test has_fit_methods(BasicStructural)
  
    model = BasicStructural(log_air_passengers, 12)
    StateSpaceModels.fit(model)
    @test loglike(model) ≈ 234.33641 atol = 1e-5 rtol = 1e-5

    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simualting
    scenarios = simulate_scenarios(model, 10, 10_000)
    test_scenarios_adequacy_with_forecast(forec, scenarios, 0.2, 0.2)
end