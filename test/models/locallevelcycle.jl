@testset "Local Level With Cycle Model" begin
    rj_temp = CSV.read(StateSpaceModels.RJ_TEMPERATURE, DataFrame).Values

    @test has_fit_methods(LocalLevelCycle)
    model = LocalLevelCycle(earth_temp_loc)
    fit!(model)

    # forecasting
    forec = forecast(model, 52)
    @test monotone_forecast_variance(forec)
end