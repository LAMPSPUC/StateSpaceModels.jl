# Tests
@testset "Strutural model tests" begin

    # @testset "Constant signal with basic structural model" begin
    #     y = ones(30)
    #     model = structuralmodel(y, 2)

    #     @test isa(model, StateSpaceModels.StateSpaceModel)
    #     @test model.mode == "time-invariant"

    #     ss = statespace(model)

    #     @test isa(ss, StateSpaceModels.StateSpace)

    #     @test all(ss.param.sqrtH .< 1e-6)
    #     @test all(ss.param.sqrtQ .< 1e-6)
    # end

    # @testset "Constant signal with exogenous variables" begin
    #     y = ones(15)
    #     X = randn(15, 2)

    #     model = structuralmodel(y, 2; X = X)

    #     @test isa(model, StateSpaceModels.StateSpaceModel)
    #     @test model.mode == "time-variant"

    #     ss = statespace(model)

    #     @test isa(ss, StateSpaceModels.StateSpace)

    #     @test all(ss.param.sqrtH .< 1e-6)
    #     @test all(ss.param.sqrtQ .< 1e-6)
    # end

    @testset "Multivariate test" begin
       
        y = [ones(20) collect(1:20)]
        model = structuralmodel(y, 2)
        ss = statespace(model)
        sim  = simulate(ss, 10, 1000)

        @test mean(sim, dims = 3)[1, :] ≈ ones(10) rtol = 1e-3
        @test mean(sim, dims = 3)[2, :] ≈ collect(21:30) rtol = 1e-3
    end

    # @testset "Air passengers" begin

    #     AP = CSV.read("../example/AirPassengers.csv")
    #     logAP = log.(Vector{Float64}(AP[:Passengers]))

    #     model = structuralmodel(logAP, 12)

    #     @test isa(model, StateSpaceModels.StateSpaceModel)
    #     @test model.mode == "time-invariant"

    #     ss = statespace(model)

    #     @test isa(ss, StateSpaceModels.StateSpace)

    # end

    # @testset "Error tests" begin

    #     y = ones(15, 1)
    #     dim = StateSpaceModels.StateSpaceDimensions(1, 1, 1, 1)
    #     Z = Vector{Matrix{Float64}}(undef, 3)
    #     T = R = Matrix{Float64}(undef, 2, 2)

    #     @test_throws ErrorException structuralmodel(y, 2; X = ones(10, 2))

    # end

end

@testset "Simulation tests" begin

    @testset "Affine series simulation" begin

        y = collect(1.:30)
        model = structuralmodel(y, 2)
        ss = statespace(model)
        sim = simulate(ss, 20, 100)
        media_sim = mean(sim, dims = 3)[1, :]

        @test size(sim) == (1, 20, 100)
        @test media_sim ≈ collect(31.:50) rtol = 1e-3
        @test var(sim[1, end, :]) < 1e-6

    end

end