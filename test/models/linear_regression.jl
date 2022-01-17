@testset "Regression" begin
    @test has_fit_methods(LinearRegression)

    # Test for 10 random points
    for _ in 1:10
        X = rand(100, 5)
        y = rand(100)
        model = LinearRegression(X, y)
        fit!(model; save_hyperparameter_distribution=false)
        β = X \ y
        for i in 1:size(X, 2)
            @test get_constrained_value(model, "β_$i") .≈ β[i] atol = 1e-5
        end
    end
    X = [
        1 1
        1 2
        1 3
        1 4
        1 5.0
    ]
    y = [
        6
        7
        8
        9
        10.0
    ]
    model = LinearRegression(X, y)
    fit!(model; save_hyperparameter_distribution=false)
    forec = forecast(model, [1 6.0])
    forec.expected_value[1] == [11.0]
end
