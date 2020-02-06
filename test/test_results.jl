AP = CSV.read("../examples/AirPassengers.csv")
logAP = log.(Vector{Float64}(AP.Passengers))

@testset "Air passengers with Kalman filter" begin
    model1 = structural(logAP, 12)

    @test isa(model1, StateSpaceModel)
    @test model1.mode == "time-invariant"
    
    ss1 = statespace(model1)
    
    @test ss1.filter_type <: KalmanFilter
    @test isa(ss1, StateSpace)
    compare_forecast_simulation(ss1, 20, 1000, 1e-2)

    model2 = structural(logAP, 12)
    
    ss2 = statespace(model2; filter_type = SquareRootFilter{Float64})
    
    @test ss2.filter_type <: SquareRootFilter
    @test isa(ss2, StateSpace)
    compare_forecast_simulation(ss2, 20, 1000, 1e-2)

    model3 = structural(logAP, 12)
    
    ss3 = statespace(model3; filter_type = UnivariateKalmanFilter{Float64})
    
    @test ss3.filter_type <: UnivariateKalmanFilter
    @test isa(ss3, StateSpace)
    compare_forecast_simulation(ss3, 20, 1000, 1e-2)

    @test ss2.smoother.alpha ≈ ss1.smoother.alpha rtol = 1e-2
    @test ss3.smoother.alpha ≈ ss1.smoother.alpha rtol = 1e-3
    @test ss2.filter.a ≈ ss1.filter.a rtol = 1e-3
    @test ss3.filter.a ≈ ss1.filter.a rtol = 1e-3

    diags1 = diagnostics(ss1)
    diags2 = diagnostics(ss2)
    diags3 = diagnostics(ss3)

    @test diags1.p_jarquebera ≈ diags2.p_jarquebera atol = 1e-2
    @test diags1.p_jarquebera ≈ diags3.p_jarquebera atol = 1e-2
    @test diags1.p_ljungbox ≈ diags2.p_ljungbox atol = 1e-2
    @test diags1.p_ljungbox ≈ diags3.p_ljungbox atol = 1e-2
    @test diags1.p_homo ≈ diags2.p_homo atol = 1e-2
    @test diags1.p_homo ≈ diags3.p_homo atol = 1e-2
end