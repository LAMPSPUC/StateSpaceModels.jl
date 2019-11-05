@testset "regression" begin
    Random.seed!(10)
    beta = [10; 20; 30]
    n = 1000
    X = rand(n, 3)
    y = X*beta + randn(n)

    model = regression(y, X)

    ss = statespace(model)

    @test ss.smoother.alpha[end, :] ≈ [10; 20; 30] rtol = 1e-2
    @test ss.model.H[1] ≈ 1.0 rtol = 1e-1
end
