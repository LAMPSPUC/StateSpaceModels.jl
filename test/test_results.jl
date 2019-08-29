AP = CSV.read("../examples/AirPassengers.csv")
logAP = log.(Vector{Float64}(AP[:Passengers]))

@testset "Air passengers with Kalman filter" begin
    model = structural(logAP, 12)

    @test isa(model, StateSpaceModel)
    @test model.mode == "time-invariant"
    
    ss1 = statespace(model)
    
    @test ss1.filter_type == KalmanFilter
    @test isa(ss1, StateSpace)
    compare_forecast_simulation(ss1, 20, 1000, 1e-2)
    
    ss2 = statespace(model; filter_type = SquareRootFilter)
    
    @test ss2.filter_type == SquareRootFilter
    @test isa(ss2, StateSpace)
    compare_forecast_simulation(ss2, 20, 1000, 1e-2)
    
    ss3 = statespace(model; filter_type = UnivariateKalmanFilter)
    
    @test ss3.filter_type == UnivariateKalmanFilter
    @test isa(ss3, StateSpace)
    compare_forecast_simulation(ss3, 20, 1000, 1e-2)

    @test ss2.smoother.alpha ≈ ss1.smoother.alpha rtol = 1e-3
    @test ss3.smoother.alpha ≈ ss1.smoother.alpha rtol = 1e-3 # !!! univariate failing
    @test ss2.filter.a ≈ ss1.filter.a rtol = 1e-3
    @test ss3.filter.a ≈ ss1.filter.a rtol = 1e-3
end