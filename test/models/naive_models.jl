@testset "Naive models" begin
    nile = CSV.File(StateSpaceModels.NILE) |> DataFrame

    model = Naive(nile.flow)
    fit!(model)
    forec = forecast(model, 10; bootstrap = true)
    forec = forecast(model, 10)
    scenarios = simulate_scenarios(model, 10, 1_000)
    @test monotone_forecast_variance(forec)
    test_scenarios_adequacy_with_forecast(forec, scenarios; rtol = 1e-1)

    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)

    model = SeasonalNaive(log_air_passengers, 12)
    fit!(model)
    forec = forecast(model, 60; bootstrap = true)
    forec = forecast(model, 60)
    scenarios = simulate_scenarios(model, 60, 1_000)
    @test monotone_forecast_variance(forec)

    # Just see if it runs
    model = ExperimentalSeasonalNaive(log_air_passengers, 12)
    fit!(model)
    forec = forecast(model, 60)
    scenarios = simulate_scenarios(model, 60, 1_000)
end