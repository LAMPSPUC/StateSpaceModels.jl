@testset "Basic Structural With Explanatory Model" begin
    @test has_fit_methods(BasicStructuralExplanatory)
    y = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    logap = log.(y.passengers)
    X = rand(length(logap), 2)
    model = BasicStructuralExplanatory(logap, 12, X)
    fit!(model)
    model.results
    # forecasting
    # For a fixed forecasting explanatory the variance must not decrease
    forec = forecast(model, ones(10, 2))
    @test monotone_forecast_variance(forec)
    kf = kalman_filter(model);
    ks = kalman_smoother(model);
    a = get_predictive_state(kf)
    @test a[1, 14] ≈ a[end, 14] atol=1e-3
    @test a[1, 15] ≈ a[end, 15] atol=1e-3
    @test_throws AssertionError simulate_scenarios(model, 10, 1000, ones(5, 2))
    scenarios = simulate_scenarios(model, 10, 1000, ones(10, 2))
    test_scenarios_adequacy_with_forecast(forec, scenarios)
end