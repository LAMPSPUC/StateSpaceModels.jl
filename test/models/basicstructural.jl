@testset "Basic Structural Model" begin
    # Univariate
    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)

    @test has_fit_methods(BasicStructural)

    model = BasicStructural(log_air_passengers, 12)
    fit!(model)
    # Runned on Python statsmodels
    @test loglike(model) ≈ 234.33641 atol = 5 rtol = 5e-2

    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simualting
    scenarios = simulate_scenarios(model, 10, 10_000)
    test_scenarios_adequacy_with_forecast(forec, scenarios)
    
    model = BasicStructural(log_air_passengers, 12)
    steadystate_tol = 1e-5
    a1 = zeros(num_states(model))
    P1 = 1e6 .* Matrix{Float64}(I, num_states(model), num_states(model))
    sparse_filter = SparseUnivariateKalmanFilter(a1, P1, num_states(model), steadystate_tol)
    fit!(model; filter=sparse_filter)
    # Runned on Python statsmodels
    @test loglike(model) ≈ 234.33641 atol = 5 rtol = 5e-2

    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simualting
    scenarios = simulate_scenarios(model, 10, 10_000)
    test_scenarios_adequacy_with_forecast(forec, scenarios)
end