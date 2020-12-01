@testset "Local Level With Explanaatory Model" begin
    @test has_fit_methods(LocalLevelExplanatory)
    y = rand(30)
    X = rand(30, 2)
    model = LocalLevelExplanatory(y, X)
    fit!(model)

    # forecasting
    # For a fixed forecasting explanatory the variance must not decrease
    forec = forecast(model, ones(5, 2))
    @test monotone_forecast_variance(forec)
    kf = kalman_filter(model)
    a = get_predictive_state(kf)
    @test a[1, 2] ≈ a[end, 2] atol=1e-3
    @test a[1, 3] ≈ a[end, 3] atol=1e-3
end
