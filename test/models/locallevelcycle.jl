@testset "Local Level With Cycle Model" begin
    rj_temp = CSV.read(StateSpaceModels.RJ_TEMPERATURE, DataFrame).Values

    @test has_fit_methods(LocalLevelCycle)
    model = LocalLevelCycle(rj_temp)
    fit!(model)

    # forecasting
    forec = forecast(model, 52)
    @test monotone_forecast_variance(forec)

    smoother = kalman_smoother(model)
    alpha = get_smoothed_state(smoother)
    @test maximum(alpha[:, 1]) >= 296
    @test minimum(alpha[:, 1]) <= 296
    @test maximum(alpha[:, 2]) >= 7.5
    @test minimum(alpha[:, 2]) <= -7.5
end