@testset "ExponentialSmoothing" begin
    @test has_fit_methods(ExponentialSmoothing)

    nile = CSV.File(StateSpaceModels.NILE) |> DataFrame
    model = ExponentialSmoothing(nile.flow)
    fit!(model)
    # statsmodels gives -638.031
    @test loglike(model) ≈ -638.025 atol = 1e-5 rtol = 1e-5
    filt = kalman_filter(model)
    smoother = kalman_smoother(model)
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    finland_fatalities = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
    log_finland_fatalities = log.(finland_fatalities.ff)
    model = ExponentialSmoothing(log_finland_fatalities; trend = true)
    fit!(model)
    @test loglike(model) ≈ 32.258 atol = 1e-5 rtol = 1e-5
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    finland_fatalities = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
    log_finland_fatalities = log.(finland_fatalities.ff)
    model = ExponentialSmoothing(log_finland_fatalities; trend = true, damped_trend = true)
    fit!(model)
    @test loglike(model) ≈ 21.000 atol = 1e-3 rtol = 1e-3
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)
    model = ExponentialSmoothing(log_air_passengers; trend = true, seasonal = 12)
    # statsmodels gives -273.215
    fit!(model)
    @test loglike(model) ≈ 275.313 atol = 1e-3 rtol = 1e-3
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    # Automatic ETS
    model = auto_ets(nile.flow)
    @test model.trend == false

    model = auto_ets(log_finland_fatalities)
    @test model.trend == true
    @test model.damped_trend == false

    model = auto_ets(log_air_passengers; seasonal = 12)
    @test model.trend == true
    @test model.damped_trend == false
end