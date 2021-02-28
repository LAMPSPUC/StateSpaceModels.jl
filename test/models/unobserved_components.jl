@testset "UnobservedComponents" begin
    @test StateSpaceModels.has_fit_methods(StateSpaceModels.UnobservedComponents)

    nile = CSV.read(StateSpaceModels.NILE, DataFrame)
    model = StateSpaceModels.UnobservedComponents(nile.flow)
    @show "fit começa $(now())"
    StateSpaceModels.fit!(model)
    @show "fit termina $(now())"
    @test StateSpaceModels.loglike(model) ≈ -632.5376 atol = 1e-5 rtol = 1e-5
    filt = StateSpaceModels.kalman_filter(model)
    smoother = StateSpaceModels.kalman_smoother(model)
    forec = StateSpaceModels.forecast(model, 10)
    @test monotone_forecast_variance(forec)

    nile = CSV.read(StateSpaceModels.NILE, DataFrame)
    model = StateSpaceModels.UnobservedComponents(nile.flow)
    @show "fit começa $(now())"
    StateSpaceModels.fit!(model)
    @show "fit termina $(now())"
    @test StateSpaceModels.loglike(model) ≈ -632.5376 atol = 1e-5 rtol = 1e-5
    filt = StateSpaceModels.kalman_filter(model)
    smoother = StateSpaceModels.kalman_smoother(model)
    forec = StateSpaceModels.forecast(model, 10)
    @test monotone_forecast_variance(forec)

    finland_fatalities = CSV.read(StateSpaceModels.VEHICLE_FATALITIES, DataFrame)
    log_finland_fatalities = log.(finland_fatalities.ff)
    model = StateSpaceModels.UnobservedComponents(log_finland_fatalities; trend = "local linear trend")
    @show "fit começa $(now())"
    StateSpaceModels.fit!(model)
    @show "fit termina $(now())"
    @test StateSpaceModels.loglike(model) ≈ 26.740 atol = 1e-5 rtol = 1e-5
    forec = StateSpaceModels.forecast(model, 10)
    @test monotone_forecast_variance(forec)

    air_passengers = CSV.read(StateSpaceModels.AIR_PASSENGERS, DataFrame)
    log_air_passengers = log.(air_passengers.passengers)
    model = StateSpaceModels.UnobservedComponents(log_air_passengers; trend = "local linear trend", seasonal = "stochastic 12")
    @show "fit começa $(now())"
    StateSpaceModels.fit!(model)
    @show "fit termina $(now())"
    @test StateSpaceModels.loglike(model) ≈ 234.33641 atol = 1e-5 rtol = 1e-5
    forec = StateSpaceModels.forecast(model, 10)
    @test monotone_forecast_variance(forec)

    rj_temp = CSV.read(StateSpaceModels.RJ_TEMPERATURE, DataFrame).Values
    model = StateSpaceModels.UnobservedComponents(rj_temp; trend = "local level", cycle = "stochastic")
    @show "fit começa $(now())"
    StateSpaceModels.fit!(model)
    @show "fit termina $(now())"
    # TODO check with other software maybe statsmodels
    @test StateSpaceModels.loglike(model) ≈ -619.8679932 atol = 1e-5 rtol = 1e-5
    filt = StateSpaceModels.kalman_filter(model)
    forec = StateSpaceModels.forecast(model, 52)
    @test monotone_forecast_variance(forec)
    smoother = StateSpaceModels.kalman_smoother(model)
    alpha = StateSpaceModels.get_smoothed_state(smoother)
    @test maximum(alpha[:, 1]) >= 296
    @test minimum(alpha[:, 1]) <= 296
    @test maximum(alpha[:, 2]) >= 7.5
    @test minimum(alpha[:, 2]) <= -7.5
end