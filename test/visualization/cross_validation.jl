@testset "Visualization Forecast" begin
    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)
    model = BasicStructural(log_air_passengers, 12)
    # forecasting
    b = cross_validation(model, 24, 110)
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), b, "str")
    @test length(rec) == 2
end