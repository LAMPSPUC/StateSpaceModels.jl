@testset "Affine series simulation" begin
    y = collect(1.:30)
    model = structural(y, 2)

    @test isa(model, StateSpaceModels.StateSpaceModel)
    @test model.mode == "time-invariant"
    @test model.filter_type == KalmanFilter

    ss = statespace(model)
    sim = simulate(ss, 20, 100)
    media_sim = mean(sim, dims = 3)[1, :]

    @test size(sim) == (1, 20, 100)
    @test media_sim ≈ collect(31.:50) rtol = 1e-3
    @test var(sim[1, end, :]) < 1e-4
end