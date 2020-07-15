@testset "Basic Structural Model" begin
    AirPassengers_dataset = readdlm(joinpath(dirname(@__DIR__()), "datasets/AirPassengers.csv"), ',')
    air_passengers = float.(AirPassengers_dataset[2:end, 2])
    log_air_passengers = log.(air_passengers)
    model = BasicStructural(log_air_passengers, 12)
    fit(model)
    @test loglike(model) ≈ 234.33641 atol = 1e-5 rtol = 1e-5
end