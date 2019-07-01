@testset "Linear trend model with Kalman filter" begin
        y = collect(1:15.0)
        unimodel = linear_trend(y)

        @test isa(unimodel, StateSpaceModels.StateSpaceModel)
        @test unimodel.mode == "time-invariant"
        @test unimodel.filter_type == KalmanFilter

        ss = statespace(unimodel)

        @test isa(ss, StateSpaceModels.StateSpace)
        @test ss.smoother.alpha[:, 2] ≈ ones(15) rtol = 1e-4
end

@testset "Linear trend model with square-root Kalman filter" begin
        y = collect(1:15.0)
        unimodel = linear_trend(y; filter_type = SquareRootFilter)

        @test isa(unimodel, StateSpaceModels.StateSpaceModel)
        @test unimodel.mode == "time-invariant"
        @test unimodel.filter_type == SquareRootFilter

        ss = statespace(unimodel)

        @test isa(ss, StateSpaceModels.StateSpace)
        @test ss.smoother.alpha[:, 2] ≈ ones(15) rtol = 1e-4
end