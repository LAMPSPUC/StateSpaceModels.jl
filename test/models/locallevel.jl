@testset "LocalLevel" begin 
    Nile_dataset = readdlm(joinpath(dirname(@__DIR__()), "datasets/Nile.csv"), ',')
    y = float.(Nile_dataset[2:end, 2])
    model = LocalLevel(y)
    fit(model)
    @test loglike(model) ≈ -632.5376 atol = 1e-5 rtol = 1e-5

    # Durbin Koopman 2012 section 2.2.5
    a1 = 0.0
    P1 = 1e7
    scalar_filter = ScalarKalmanFilter(a1, P1, 1)
    model = LocalLevel(y)
    fit(model; filter = scalar_filter)

    # Without the concentrated filter and score calculation this is close enough
    @test get_constrained_value(model, "sigma2_ε") ≈ 15099 atol = 2
    @test get_constrained_value(model, "sigma2_η") ≈ 1469.1 atol = 2
end
