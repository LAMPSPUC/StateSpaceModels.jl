# Tests
@testset "Strutural model tests" begin

    @testset "Constant signal test" begin
        y = ones(15)
        model = StructuralModel(y, 17)

        @test isa(model, StateSpaceModels.StateSpaceModel)
        @test isa(model, StateSpaceModels.BasicStructuralModel)

        ss = statespace(model; nseeds = 3)

        @test all(ss.param.sqrtH .< 1e-6)
        @test all(ss.param.sqrtQ .< 1e-6)
    end

end
