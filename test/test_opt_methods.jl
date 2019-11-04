@testset "LBFGS" begin
    model = local_level(rand(1000))

    # Test error
    opt_method = LBFGS(model, 10)
    @test opt_method.n_seeds == 10
    @test length(opt_method.seeds) == 10

    opt_method = LBFGS(model, [[1.0; 2.0]])
    @test opt_method.n_seeds == 1
    @test opt_method.seeds == [[1.0; 2.0]]

    err = ErrorException("Seed 1 has 1 elements and the model has 2 unknowns.")
    @test_throws err LBFGS(model, [[1.0]])
end

@testset "BFGS" begin
    model = local_level(rand(1000))

    # Test error
    opt_method = BFGS(model, 10)
    @test opt_method.n_seeds == 10
    @test length(opt_method.seeds) == 10

    opt_method = BFGS(model, [[1.0; 2.0]])
    @test opt_method.n_seeds == 1
    @test opt_method.seeds == [[1.0; 2.0]]

    err = ErrorException("Seed 1 has 1 elements and the model has 2 unknowns.")
    @test_throws err BFGS(model, [[1.0]])
end