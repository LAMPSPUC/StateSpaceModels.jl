@testset "Affine series simulation" begin
    y = collect(1.:30)
    model = structural(y, 2)

    @test isa(model, StateSpaceModel)
    @test model.mode == "time-invariant"
    
    ss = statespace(model)
    @test ss.filter_type <: KalmanFilter

    sim = simulate(ss, 20, 100)
    media_sim = mean(sim, dims = 3)[:, 1]

    @test size(sim) == (20, 1, 100)
    @test media_sim ≈ collect(31.:50) rtol = 1e-3
    @test var(sim[1, end, :]) < 1e-4
    compare_forecast_simulation(ss, 20, 1000, 1e-3)
end