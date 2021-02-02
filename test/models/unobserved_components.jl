@testset "UnobservedComponents" begin
    @test has_fit_methods(UnobservedComponents)

    nile = CSV.read(StateSpaceModels.NILE, DataFrame)
    model = UnobservedComponents(nile.flow)
    fit!(model)
    @test loglike(model) ≈ -632.5376 atol = 1e-5 rtol = 1e-5
    filt = kalman_filter(model)
    smoother = kalman_smoother(model)
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    finland_fatalities = CSV.read(StateSpaceModels.VEHICLE_FATALITIES, DataFrame)
    log_finland_fatalities = log.(finland_fatalities.ff)
    model = UnobservedComponents(log_finland_fatalities; trend = "local linear trend")
    fit!(model)
    @test loglike(model) ≈ 26.740 atol = 1e-5 rtol = 1e-5
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    air_passengers = CSV.read(StateSpaceModels.AIR_PASSENGERS, DataFrame)
    log_air_passengers = log.(air_passengers.passengers)
    model = UnobservedComponents(log_air_passengers; trend = "local linear trend", seasonal = "stochastic 12")
    fit!(model)
    @test loglike(model) ≈ 234.33641 atol = 1e-5 rtol = 1e-5
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    rj_temp = CSV.read(StateSpaceModels.RJ_TEMPERATURE, DataFrame).Values
    model = UnobservedComponents(rj_temp; trend = "local level", cycle = "stochastic")
    fit!(model)
    # TODO check with other software maybe statsmodels
    @test loglike(model) ≈ -619.8679932 atol = 1e-5 rtol = 1e-5
    filt = kalman_filter(model)
    forec = forecast(model, 52)
    @test monotone_forecast_variance(forec)
    smoother = kalman_smoother(model)
    alpha = get_smoothed_state(smoother)
    @test maximum(alpha[:, 1]) >= 296
    @test minimum(alpha[:, 1]) <= 296
    @test maximum(alpha[:, 2]) >= 7.5
    @test minimum(alpha[:, 2]) <= -7.5
end