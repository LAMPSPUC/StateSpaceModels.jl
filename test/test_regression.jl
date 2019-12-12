@testset "regression" begin
    Random.seed!(14)
    beta = [10; 20; 30]
    n = 1000
    X = rand(n, 3)
    y = X*beta + randn(n)

    model = regression(y, X)

    ss = statespace(model)

    beta_OLS = X\y

    # The result should be the same of OLS
    @test ss.filter.att[end, :] ≈ beta_OLS rtol = 1e-4
    @test ss.model.H[1] ≈ 1.0 rtol = 1e-2
end
