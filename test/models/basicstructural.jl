@testset "Basic Structural Model" begin
    log_air_passengers = log.(StateSpaceModels.AIRPASSENGERS)
    model = BasicStructural(log_air_passengers, 12)
    fit(model)
    @test loglike(model) ≈ 234.33641 atol = 1e-5 rtol = 1e-5
end