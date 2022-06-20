@testset "Visualization Diagnostics" begin
    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)
    model = BasicStructural(log_air_passengers, 12)
    fit!(model)
    kf = kalman_filter(model)
    @test typeof(plotdiagnostics) <: Function
    @test length(methods(plotdiagnostics)) == 1
    @test typeof(plotdiagnostics!) <: Function
    @test length(methods(plotdiagnostics!)) == 2
end