@testset "Visualization Unobserved Components" begin
    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)
    model = UnobservedComponents(log_air_passengers; trend = "local linear trend", seasonal = "stochastic 12")
    fit!(model)
    kf = kalman_filter(model)
    ks = kalman_smoother(model)

    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, kf)
    @test length(rec) == 4
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, ks)
    @test length(rec) == 4

    finland_fatalities = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
    log_finland_fatalities = log.(finland_fatalities.ff)
    model = UnobservedComponents(log_finland_fatalities; trend = "local linear trend")
    fit!(model)

    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, kf)
    @test length(rec) == 3
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, ks)
    @test length(rec) == 3

    model = ExponentialSmoothing(log_air_passengers; trend = true, seasonal = 12)
    fit!(model)
    kf = kalman_filter(model)
    ks = kalman_smoother(model)

    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, kf)
    @test length(rec) == 4
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, ks)
    @test length(rec) == 4

    finland_fatalities = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
    log_finland_fatalities = log.(finland_fatalities.ff)
    model = ExponentialSmoothing(log_finland_fatalities; trend = true)
    fit!(model)

    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, kf)
    @test length(rec) == 3
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, ks)
    @test length(rec) == 3
end