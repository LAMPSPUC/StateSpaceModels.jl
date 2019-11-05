<<<<<<< HEAD
@testset "regresison" begin
    Random.seed!(14)
=======
@testset "regression" begin
    Random.seed!(10)
>>>>>>> 2bcbafe7b31f9065b1342ff9eadb4ebaa699ab42
    beta = [10; 20; 30]
    n = 1000
    X = rand(n, 3)
    y = X*beta + randn(n)

    model = regression(y, X)

    ss = statespace(model)

    @test ss.smoother.alpha[end, :] ≈ [10; 20; 30] rtol = 1e-2
<<<<<<< HEAD
    @test ss.model.H[1] ≈ 1.0 rtol = 1e-2
end
=======
    @test ss.model.H[1] ≈ 1.0 rtol = 1e-1
end
>>>>>>> 2bcbafe7b31f9065b1342ff9eadb4ebaa699ab42
