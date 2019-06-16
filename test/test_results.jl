# Here we should put tests that obtained the same results in other softwares

@testset "Air passengers" begin

    AP = CSV.read("../example/AirPassengers.csv")
    logAP = log.(Vector{Float64}(AP[:Passengers]))

    model = structuralmodel(logAP, 12)

    @test isa(model, StateSpaceModels.StateSpaceModel)
    @test model.mode == "time-invariant"

    ss = statespace(model)
    ss.filter

    @test isa(ss, StateSpaceModels.StateSpace)
    # We should test what is the covariance it returns
end