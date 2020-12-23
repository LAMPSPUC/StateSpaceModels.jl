@testset "UnobservedComponents" begin
    @test has_fit_methods(UnobservedComponents)

    nile = CSV.read(StateSpaceModels.NILE, DataFrame)
    model = UnobservedComponents(nile.flow)
    fit!(model)
    @test loglike(model) ≈ -632.5376 atol = 1e-5 rtol = 1e-5

    finland_fatalities = CSV.read(StateSpaceModels.VEHICLE_FATALITIES, DataFrame)
    log_finland_fatalities = log.(finland_fatalities.ff)
    model = UnobservedComponents(log_finland_fatalities; trend = "local linear trend")
    fit!(model)
    # The result is close but not correct because of initial_hyperparameters
    @test_broken StateSpaceModels.loglike(model) ≈ 26.740 atol = 1e-5 rtol = 1e-5

    air_passengers = CSV.read(StateSpaceModels.AIR_PASSENGERS, DataFrame)
    log_air_passengers = log.(air_passengers.passengers)
    model = UnobservedComponents(log_air_passengers; trend = "local linear trend", seasonal = "stochastic 12")
    fit!(model)
    @test loglike(model) ≈ 234.33641 atol = 1e-5 rtol = 1e-5
end