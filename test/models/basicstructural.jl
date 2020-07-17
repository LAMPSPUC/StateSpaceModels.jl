@testset "Basic Structural Model" begin
    air_passengers = read_csv(StateSpaceModels.AIRPASSENGERS)
    log_air_passengers = log.(air_passengers.passengers)

    @assert is_valid_statespacemodel(BasicStructural)
    model = BasicStructural(log_air_passengers, 12)
    fit(model)
    @test loglike(model) ≈ 234.33641 atol = 1e-5 rtol = 1e-5
end