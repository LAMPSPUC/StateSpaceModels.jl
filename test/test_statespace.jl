# Tests
@testset "Strutural model tests" begin

    @testset "Constant signal with basic structural model" begin
        y = ones(15)
        model = structuralmodel(y, 2)

        @test isa(model, StateSpaceModels.StateSpaceModel)

        ss = statespace(model)

        @test isa(ss, StateSpaceModels.StateSpace)

        @test all(ss.param.sqrtH .< 1e-6)
        @test all(ss.param.sqrtQ .< 1e-6)
    end

    @testset "Constant signal with exogenous variables" begin
        y = ones(15)
        X = randn(15, 2)

        model = structuralmodel(y, 2; X = X)

        @test isa(model, StateSpaceModels.StateSpaceModel)

        ss = statespace(model)

        @test isa(ss, StateSpaceModels.StateSpace)

        @test all(ss.param.sqrtH .< 1e-6)
        @test all(ss.param.sqrtQ .< 1e-6)
    end

    @testset "Air passengers" begin

        AP = CSV.read("../example/AirPassengers.csv")
        logAP = log.(Vector{Float64}(AP[:Passengers]))

        model = structuralmodel(logAP, 12)
        ss = statespace(model)

        @test isa(model, StateSpaceModels.StateSpaceModel)
        @test isa(ss, StateSpaceModels.StateSpace)

    end

end