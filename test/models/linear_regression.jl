@testset "Regression" begin
    # Test for 10 random points
    for _ in 1:10
        X = rand(100, 5)
        y = rand(100)
        model = LinearRegression(X, y)
        fit(model)
        β = X\y
        for i in 1:size(X, 2)
            @test get_constrained_value(model, "β_$i") .≈ β[i] atol = 1e-5
        end
    end
end