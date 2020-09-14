@testset "LocalLevel" begin
    nile = CSV.read(StateSpaceModels.NILE, DataFrame)

    @test has_fit_methods(LocalLevel)

    model = LocalLevel(nile.flow)
    @test_throws ErrorException show(stdout, results(model))
    fit!(model)
    result_print = 
    """                         Results                        
    ========================================================
    Number of observations:       100
    Number of unknown parameters: 2
    Log-likelihood:               -632.54
    AIC:                          1269.08
    BIC:                          1274.29
    --------------------------------------------------------
    Parameter      Estimate   Std.Error     z stat   p-value
    sigma2_ε       15108.33      164.21      92.01      0.00
    sigma2_η        1463.16      279.88       5.23      0.00
    """
    @test sprint(show, results(model)) == result_print
    @test loglike(model) ≈ -632.5376 atol = 1e-5 rtol = 1e-5

    # Durbin Koopman 2012 section 2.2.5
    a1 = 0.0
    P1 = 1e7
    scalar_filter = ScalarKalmanFilter(a1, P1, 1)
    model = LocalLevel(nile.flow)
    fit!(model; filter=scalar_filter)

    # Without the concentrated filter and score calculation this is close enough
    @test get_constrained_value(model, "sigma2_ε") ≈ 15099 rtol = 1e-3
    @test get_constrained_value(model, "sigma2_η") ≈ 1469.1 rtol = 1e-3

    # Fix some parameters
    fix_hyperparameters!(model, Dict("sigma2_ε" => 15099.0))
    fit!(model; filter=scalar_filter)
    result_print = 
    """                         Results                        
    ========================================================
    Number of observations:       100
    Number of unknown parameters: 1
    Log-likelihood:               -632.54
    AIC:                          1267.09
    BIC:                          1269.69
    --------------------------------------------------------
    Parameter      Estimate   Std.Error     z stat   p-value
    sigma2_ε       15099.00          -          -         - 
    sigma2_η        1468.29      175.18       8.38      0.00
    """
    @test sprint(show, results(model)) == result_print

    hyperparameters = get_hyperparameters(model)
    @test !isempty(get_minimizer_hyperparameter_position(hyperparameters))
    
    @test loglike(model; filter=scalar_filter) ≈  -632.54421 atol = 1e-5 rtol = 1e-5
    @test get_constrained_value(model, "sigma2_η") ≈ 1469.1 atol = 1

    # Estimate with Float32
    nile32 = Float32.(nile.flow)
    model = LocalLevel(nile32)
    fit!(model)
    @test loglike(model) ≈ -632.53766f0 atol = 1e-5 rtol = 1e-5

    # Missing values 
    nile.flow[[collect(21:40); collect(61:80)]] .= NaN
    model = LocalLevel(nile.flow)
    fit!(model)
    @test loglike(model) ≈ -379.9899 atol = 1e-5 rtol = 1e-5

    filter = kalman_filter(model)
    smoother = kalman_smoother(model)
    @test filter.Ptt[end] == smoother.V[end] # by construction

    for t in 2:length(model.system.y) - 1
        @test filter.Ptt[t][1] > smoother.V[t][1] # by construction
    end

    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simulating
    scenarios = simulate_scenarios(model, 10, 100_000)
    test_scenarios_adequacy_with_forecast(forec, scenarios)

end
