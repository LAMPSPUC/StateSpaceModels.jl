@testset "Prints" begin
    random_series = rand(100)
    # Default show methods
    model = LocalLevel(random_series)
    show(model)
    println("")
    fit!(model)
    show(results(model))
    println("")
    # Specific show methods
    model = SARIMA(random_series)
    show(model)
end