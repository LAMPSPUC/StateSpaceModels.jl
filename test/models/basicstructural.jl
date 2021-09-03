@testset "Basic Structural Model" begin
    # Univariate
    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)

    @test has_fit_methods(BasicStructural)

    model = BasicStructural(log_air_passengers, 12)
    fit!(model)
    # Runned on Python statsmodels
    @test loglike(model) ≈ 234.33641 atol = 1e-5 rtol = 1e-5

    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simualting
    scenarios = simulate_scenarios(model, 10, 10_000)
    test_scenarios_adequacy_with_forecast(forec, scenarios)

    log_ap32 = Float32.(log_air_passengers)
    model = BasicStructural(log_ap32, 12)
    @test model.system.Z == Float32[1.0; 0.0; 1.0; zeros(10)]
    @test_broken fit!(model)
    @test_broken loglike(model) ≈ 234.33641f0 atol = 1e-5 rtol = 1e-5

    
    model = BasicStructural(log_air_passengers, 12)
    steadystate_tol = 1e-5
    a1 = zeros(num_states(model))
    P1 = 1e6 .* Matrix{Float64}(I, num_states(model), num_states(model))
    sparse_filter = SparseUnivariateKalmanFilter(a1, P1, num_states(model), steadystate_tol)
    fit!(model; filter=sparse_filter)
    # Runned on Python statsmodels
    @test loglike(model) ≈ 234.33641 atol = 1e-5 rtol = 1e-5

    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simualting
    scenarios = simulate_scenarios(model, 10, 10_000)
    test_scenarios_adequacy_with_forecast(forec, scenarios)
end