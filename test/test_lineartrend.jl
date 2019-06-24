@testset "Linear trend model" begin
        y = collect(1:15.0)
        unimodel = linear_trend(y)

        @test isa(unimodel, StateSpaceModels.StateSpaceModel)
        @test unimodel.mode == "time-invariant"

        ss = statespace(unimodel)

        @test isa(ss, StateSpaceModels.StateSpace)
        @test ss.smoother.alpha[:, 2] == ones(15)
end