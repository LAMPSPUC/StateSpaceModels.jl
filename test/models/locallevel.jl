using CSV, DataFrames

@testset "LocalLevel" begin
    nile = CSV.File(StateSpaceModels.NILE) |> DataFrame

    @test has_fit_methods(LocalLevel)

    model = LocalLevel(nile.flow)

    # Test that getter functions throw error for model that hasn't been fitted yet
    @test_throws ErrorException get_innovations(model)
    @test_throws ErrorException get_innovations_variance(model)
    @test_throws ErrorException get_filtered_state(model)
    @test_throws ErrorException get_filtered_state_variance(model)
    @test_throws ErrorException get_predictive_state(model)
    @test_throws ErrorException get_predictive_state_variance(model)

    fit!(model)
    @test loglike(model) ≈ -632.5376 atol = 1e-5 rtol = 1e-5

    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simulating
    scenarios = simulate_scenarios(model, 10, 100_000)
    test_scenarios_adequacy_with_forecast(forec, scenarios)

    filter = kalman_filter(model)

    # Test that getter functions now work since model has been fitted
    @test get_innovations(model) == get_innovations(filter)
    @test get_innovations_variance(model) == get_innovations_variance(filter)
    @test get_filtered_state(model) == get_filtered_state(filter)
    @test get_filtered_state_variance(model) == get_filtered_state_variance(filter)
    @test get_predictive_state(model) == get_predictive_state(filter)
    @test get_predictive_state_variance(model) == get_predictive_state_variance(filter)

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
    model = LocalLevel(nile.flow)
    fix_hyperparameters!(model, Dict("sigma2_ε" => 15099.0))
    fit!(model; filter=scalar_filter)

    hyperparameters = get_hyperparameters(model)
    @test !isempty(hyperparameters.minimizer_hyperparameter_position)

    @test loglike(model; filter=scalar_filter) ≈ -632.54421 atol = 1e-5 rtol = 1e-5
    @test get_constrained_value(model, "sigma2_η") ≈ 1469.1 rtol = 1e-2

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
    @test filter.Ptt[end] ≈ smoother.V[end] atol = 1e-7 # by construction

    for t in 2:(length(model.system.y) - 1)
        @test filter.Ptt[t][1] > smoother.V[t][1] # by construction
    end

    # Forecasting of series with missing values
    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simulating
    scenarios = simulate_scenarios(model, 10, 100_000)
    test_scenarios_adequacy_with_forecast(forec, scenarios)
end
