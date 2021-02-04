@testset "Visualization" begin
    air_passengers = CSV.read(StateSpaceModels.AIR_PASSENGERS, DataFrame)
    log_air_passengers = log.(air_passengers.passengers)
    model = BasicStructural(log_air_passengers, 12)
    fit!(model)
    # forecasting
    forec = forecast(model, 12)
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, forec)
    @test length(rec) == 4
    # simulating
    scen = simulate_scenarios(model, 12, 100)
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), model, scen)
    @test length(rec) == 2
end