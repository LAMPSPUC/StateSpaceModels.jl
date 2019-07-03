@testset "Linear trend model with Kalman filter" begin
        y = collect(1:15.0)
        unimodel = linear_trend(y)

        @test isa(unimodel, StateSpaceModel)
        @test unimodel.mode == "time-invariant"
        
        ss = statespace(unimodel)
        
        @test ss.filter_type == KalmanFilter
        @test isa(ss, StateSpace)
        @test ss.smoother.alpha[:, 2] ≈ ones(15) rtol = 1e-4
        compare_forecast_simulation(ss, 20, 1000, 1e-3)
end

@testset "Linear trend model with square-root Kalman filter" begin
        y = collect(1:15.0)
        unimodel = linear_trend(y)

        @test isa(unimodel, StateSpaceModel)
        @test unimodel.mode == "time-invariant"
        
        ss = statespace(unimodel; filter_type = SquareRootFilter)
        
        @test ss.filter_type == SquareRootFilter
        @test isa(ss, StateSpace)
        @test ss.smoother.alpha[:, 2] ≈ ones(15) rtol = 1e-4
        compare_forecast_simulation(ss, 20, 1000, 1e-3)
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