@testset "Linear trend" begin
    @testset "Linear trend model" begin
        y = collect(1:15.0)
        unimodel = linear_trend(y)

        @test isa(unimodel, StateSpaceModel)
        @test unimodel.mode == "time-invariant"
                
        ss1 = statespace(unimodel)

        @test ss1.filter_type == KalmanFilter
        @test isa(ss1, StateSpace)
        @test ss1.smoother.alpha[:, 2] ≈ ones(15) rtol = 1e-4
        compare_forecast_simulation(ss1, 20, 1000, 1e-3)
                
        ss2 = statespace(unimodel; filter_type = SquareRootFilter)

        @test ss2.filter_type == SquareRootFilter
        @test isa(ss2, StateSpace)
        @test ss2.smoother.alpha[:, 2] ≈ ones(15) rtol = 1e-4
        compare_forecast_simulation(ss2, 20, 1000, 1e-3)
                
        ss3 = statespace(unimodel; filter_type = UnivariateKalmanFilter)

        @test ss3.filter_type == UnivariateKalmanFilter
        @test isa(ss3, StateSpace)
        @test ss3.smoother.alpha[:, 2] ≈ ones(15) rtol = 1e-4
        compare_forecast_simulation(ss3, 20, 1000, 1e-3)

        @test ss2.smoother.alpha ≈ ss1.smoother.alpha rtol = 1e-3
        @test ss3.smoother.alpha ≈ ss1.smoother.alpha rtol = 1e-3
        @test ss2.filter.a ≈ ss1.filter.a rtol = 1e-3
        @test ss3.filter.a ≈ ss1.filter.a rtol = 1e-3
    end

    @testset "Linear trend model with missing values" begin
        y = collect(1:30.0)
        y[4:8] .= NaN
        model = linear_trend(y)
        ss = statespace(model)

        @test y[9:end] ≈ ss.smoother.alpha[9:end, 1] rtol = 1e-3
        @test sum(ss.filter.P[:, :, 8]) > sum(ss.filter.P[:, :, 7]) > sum(ss.filter.P[:, :, 6]) > 
                sum(ss.filter.P[:, :, 5]) > sum(ss.filter.P[:, :, 4])
    end
end