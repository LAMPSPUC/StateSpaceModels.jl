@testset "Naive models" begin
    nile = CSV.File(StateSpaceModels.NILE) |> DataFrame

    model = Naive(nile.flow)
    fit!(model)
    StateSpaceModels.get_standard_residuals(model)
    forec = forecast(model, 10; bootstrap = true)
    forec = forecast(model, 10)
    scenarios = simulate_scenarios(model, 10, 1_000)
    @test monotone_forecast_variance(forec)
    test_scenarios_adequacy_with_forecast(forec, scenarios; rtol = 1e-1)
    cross_validation(model, 10, 70; n_scenarios=100)

    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)

    model = SeasonalNaive(log_air_passengers, 12)
    fit!(model)
    forec = forecast(model, 60; bootstrap = true)
    forec = forecast(model, 60)
    scenarios = simulate_scenarios(model, 60, 1_000)
    @test monotone_forecast_variance(forec)
    StateSpaceModels.reinstantiate(model, model.y)

    # Just see if it runs
    model = ExperimentalSeasonalNaive(log_air_passengers, 12)
    fit!(model)
    forec = forecast(model, 60)
    scenarios = simulate_scenarios(model, 60, 1_000)
    StateSpaceModels.reinstantiate(model, model.y)

    using StateSpaceModels
    using Test
    y = randn(100)
    y[10] = NaN
    model = Naive(y)
    @test_throws ErrorException fit!(model)
end