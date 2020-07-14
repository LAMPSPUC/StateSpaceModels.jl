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
    @test get_constrained_value(model, "sigma2_η") ≈ 1469.1 atol = 1

    # Fix some parameters
    fix_hyperparameters!(model, Dict("sigma2_ε" => 15099.0))
    fit(model; filter = scalar_filter)

    hyperparameters = get_hyperparameters(model)
    @test !isempty(get_minimizer_hyperparameter_position(hyperparameters))
    
    @test loglike(model; filter = scalar_filter) ≈  -632.54421 atol = 1e-5 rtol = 1e-5
    @test get_constrained_value(model, "sigma2_η") ≈ 1469.1 atol = 1

    # Estimate with Float32
    y = Float32.(Nile_dataset[2:end, 2])
    model = LocalLevel(y)
    fit(model)
    @test loglike(model) ≈ -632.53766f0 atol = 1e-5 rtol = 1e-5
end
