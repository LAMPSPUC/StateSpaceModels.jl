@testset "Simulation tests" begin
    @testset "Affine series simulation" begin
        y = collect(1.:30)
        model = structural(y, 2)
        ss = statespace(model)
        sim = simulate(ss, 20, 100)
        media_sim = mean(sim, dims = 3)[1, :]
        @test size(sim) == (1, 20, 100)
        @test media_sim ≈ collect(31.:50) rtol = 1e-3
        @test var(sim[1, end, :]) < 1e-6
    end
end