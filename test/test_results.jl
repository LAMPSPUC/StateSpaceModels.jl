@testset "Air passengers with Kalman filter" begin

    AP = CSV.read("./examples/AirPassengers.csv")
    logAP = log.(Vector{Float64}(AP[:Passengers]))

    model = structural(logAP, 12)

    @test isa(model, StateSpaceModel)
    @test model.mode == "time-invariant"
    
    ss = statespace(model)
    
    @test ss.filter_type == KalmanFilter
    @test isa(ss, StateSpace)
    compare_forecast_simulation(ss, 20, 1000, 1e-2)
end

@testset "Air passengers with square-root Kalman filter" begin

    AP = CSV.read("./examples/AirPassengers.csv")
    logAP = log.(Vector{Float64}(AP[:Passengers]))

    model = structural(logAP, 12)

    @test isa(model, StateSpaceModel)
    @test model.mode == "time-invariant"
    
    ss = statespace(model; filter_type = SquareRootFilter)
    
    @test ss.filter_type == SquareRootFilter
    @test isa(ss, StateSpace)
    compare_forecast_simulation(ss, 20, 1000, 1e-2)
end