@testset "SARIMA" begin
    internet = CSV.File(StateSpaceModels.INTERNET) |> DataFrame
    dinternet = internet.dinternet[2:end]
    @test has_fit_methods(SARIMA)

    model = SARIMA(dinternet; order = (1, 0, 1))
    fit!(model)
    @test loglike(model) ≈ -254.149 atol = 1e-5 rtol = 1e-5

    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # Prediction from Pyhton statsmodels
    predicted_mean = [
        -1.11949821,
        -0.72809028,
        -0.47352952,
        -0.30797034,
        -0.20029528,
        -0.13026644,
        -0.08472164,
        -0.05510058,
        -0.03583588,
        -0.02330665
    ]
    @test predicted_mean ≈ vcat(forec.expected_value...) atol = 1e-3
    # simualting
    scenarios = simulate_scenarios(model, 10, 30_000)
    # Values are very close to 0.0 so we test with absolute tolerance
    # It attains 1e-3 when we make 10M simulations, which is too much
    # computation for a rather simple test.
    test_scenarios_adequacy_with_forecast(forec, scenarios; atol=2e-1)

    missing_obs = [6, 16, 26, 36, 46, 56, 66, 72, 73, 74, 75, 76, 86, 96]
    missing_dinternet = copy(dinternet)
    missing_dinternet[missing_obs] .= NaN

    model = SARIMA(missing_dinternet; order = (1, 0, 1))
    fit!(model)
    @test loglike(model) ≈ -225.770 atol = 1e-5 rtol = 1e-5

    wholesale = CSV.File(StateSpaceModels.WHOLESALE_PRICE_INDEX) |> DataFrame
    model = SARIMA(wholesale.wpi; order = (1, 1, 1))
    fit!(model)
    @test loglike(model) ≈ -137.246818 atol = 1e-5 rtol = 1e-5

    uschange_consumption = CSV.File(StateSpaceModels.US_CHANGE) |> DataFrame
    model = SARIMA(uschange_consumption.Consumption; order = (1, 0, 3), include_mean = true)
    fit!(model)
    @test loglike(model) ≈ -164.8 atol = 1e-1 rtol = 1e-1

    model = SARIMA(uschange_consumption.Consumption; order = (3, 0, 0), include_mean = true)
    fit!(model)
    @test loglike(model) ≈ -165.2 atol = 1e-1 rtol = 1e-1
    
    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)
    model = SARIMA(log_air_passengers; order = (2, 1, 0), seasonal_order = (1, 1, 0, 12))
    fit!(model)
    @test_broken loglike(model) ≈ 240.821 atol = 1e-3 rtol = 1e-3

    model = SARIMA(log_air_passengers; order = (0, 1, 1), seasonal_order = (0, 1, 1, 12))
    fit!(model)
    @test loglike(model) ≈ 244.686 atol = 1e-3 rtol = 1e-3

    model = SARIMA(log_air_passengers; order = (2, 1, 1))
    fit!(model)
    @test loglike(model) ≈ 129.732 atol = 1e-3 rtol = 1e-3

    model = SARIMA(log_air_passengers; order = (2, 1, 0), seasonal_order = (1, 1, 0, 4))
    fit!(model)
    @test_broken loglike(model) ≈ 69.931 atol = 1e-3 rtol = 1e-3

    model = SARIMA(log_air_passengers; order = (0, 1, 1), seasonal_order = (0, 1, 1, 12))
    fit!(model)
    @test loglike(model) ≈ 244.696 atol = 1e-3 rtol = 1e-3

    model = SARIMA(log_air_passengers; order = (0, 1, 2), seasonal_order = (0, 1, 1, 12))
    fit!(model)
    @test loglike(model) ≈ 244.805 atol = 1e-3 rtol = 1e-3

    model = SARIMA(log_air_passengers; order = (0, 1, 2), seasonal_order = (0, 1, 2, 12))
    fit!(model)
    @test loglike(model) ≈ 245.074 atol = 1e-3 rtol = 1e-3

    model = SARIMA(log_air_passengers; order = (0, 1, 2), seasonal_order = (0, 1, 2, 5))
    fit!(model)
    @test loglike(model) ≈ 115.009 atol = 1e-3 rtol = 1e-3

    model = SARIMA(log_air_passengers; order = (1, 1, 2), seasonal_order = (0, 1, 2, 5))
    fit!(model)
    @test loglike(model) ≈ 120.128 atol = 1e-3 rtol = 1e-3

    model = SARIMA(log_air_passengers; order = (2, 0, 0), seasonal_order = (0, 1, 0, 12))
    fit!(model)
    @test loglike(model) ≈ 228.502 atol = 1e-3 rtol = 1e-3

    # auto arima tests
    model = auto_arima(dinternet)
    @test model.order.p == 1
    @test model.order.q == 1
    @test model.include_mean == false

    model = auto_arima(dinternet; max_q = 0)
    @test model.order.q == 0

    model = auto_arima(dinternet; max_q = 0, max_p = 0)
    @test model.order.p == 0
    @test model.order.q == 0

    nile = CSV.File(StateSpaceModels.NILE) |> DataFrame
    model = auto_arima(nile.flow; d = 1, show_trace = true)
    @test model.order.p == 1
    @test model.order.d == 1
    @test model.order.q == 1
    @test model.include_mean == false

    model = auto_arima(uschange_consumption.Consumption)
    @test model.order.p == 1
    @test model.order.d == 0
    @test model.order.q == 3
    @test model.include_mean == true

    model = auto_arima(float(air_passengers.passengers); seasonal = 12)
    @test model.order.d == 1
    @test model.order.D == 1

    model = auto_arima(log.(air_passengers.passengers); seasonal = 12, show_trace = true)
    @test model.order.d == 0
    @test model.order.D == 1
end