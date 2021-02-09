@testset "Visualization Unobserved Components" begin
    air_passengers = CSV.read(StateSpaceModels.AIR_PASSENGERS, DataFrame)
    log_air_passengers = log.(air_passengers.passengers)
    model = UnobservedComponents(log_air_passengers; trend = "local linear trend", seasonal = "stochastic 12")
    fit!(model)
    kf = kalman_filter(model)
    ks = kalman_smoother(model)

    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, kf)
    @test length(rec) == 3
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, ks)
    @test length(rec) == 3

    finland_fatalities = CSV.read(StateSpaceModels.VEHICLE_FATALITIES, DataFrame)
    log_finland_fatalities = log.(finland_fatalities.ff)
    model = UnobservedComponents(log_finland_fatalities; trend = "local linear trend")
    fit!(model)

    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, kf)
    @test length(rec) == 2
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, ks)
    @test length(rec) == 2
end