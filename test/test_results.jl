# Here we should put tests that obtained the same results in other softwares

@testset "Air passengers with Kalman filter" begin

    AP = CSV.read("../example/AirPassengers.csv")
    logAP = log.(Vector{Float64}(AP[:Passengers]))

    model = structural(logAP, 12)

    @test isa(model, StateSpaceModels.StateSpaceModel)
    @test model.mode == "time-invariant"
    @test model.filter_type == KalmanFilter

    ss = statespace(model)

    @test isa(ss, StateSpaceModels.StateSpace)
    compare_forecast_simulation(ss, 20, 1000, 1e-2)
end

@testset "Air passengers with square-root Kalman filter" begin

    AP = CSV.read("../example/AirPassengers.csv")
    logAP = log.(Vector{Float64}(AP[:Passengers]))

    model = structural(logAP, 12; filter_type = SquareRootFilter)

    @test isa(model, StateSpaceModels.StateSpaceModel)
    @test model.mode == "time-invariant"
    @test model.filter_type == SquareRootFilter

    ss = statespace(model)

    @test isa(ss, StateSpaceModels.StateSpace)
    compare_forecast_simulation(ss, 20, 1000, 1e-2)
end