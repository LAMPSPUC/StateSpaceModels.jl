# Tests
@testset "Strutural model tests" begin

    @testset "Constant signal with basic structural model" begin
        y = ones(15)
        model = StructuralModel(y, 2)

        @test isa(model, StateSpaceModels.StateSpaceModel)
        @test isa(model, StateSpaceModels.BasicStructuralModel)

        ss = statespace(model; nseeds = 3)

        @test all(ss.param.sqrtH .< 1e-6)
        @test all(ss.param.sqrtQ .< 1e-6)
    end

    @testset "Constant signal with exogenous variables" begin
        y = ones(15)
        X = randn(15, 2)

        model = StructuralModel(y, 2; X = X)

        @test isa(model, StateSpaceModels.StateSpaceModel)
        @test isa(model, StateSpaceModels.StructuralModelExogenous)

        ss = statespace(model; nseeds = 3)

        @test all(ss.param.sqrtH .< 1e-6)
        @test all(ss.param.sqrtQ .< 1e-6)
    end

end
