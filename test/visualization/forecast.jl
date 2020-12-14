using Plots

@testset "Visualization" begin
    air_passengers = CSV.read(StateSpaceModels.AIR_PASSENGERS, DataFrame)
    log_air_passengers = log.(air_passengers.passengers)
    model = BasicStructural(log_air_passengers, 12)
    fit!(model)
    # forecasting
    forec = forecast(model, 12)
    plot(model, forec)
end