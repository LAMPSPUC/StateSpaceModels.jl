@testset "Basic Structural Model" begin
    # Univariate
    air_passengers = CSV.read(StateSpaceModels.AIR_PASSENGERS, DataFrame)
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

    # Multivariate
    front_rear_df = CSV.read(StateSpaceModels.FRONT_REAR_SEAT_KSI, DataFrame)
    front_rear = [front_rear_df.front_seat_KSI front_rear_df.rear_seats_KSI]
  
    model = BasicStructural(front_rear, 12)
    fit!(model)
    # model.results
    # filter = StateSpaceModels.kalman_filter(model)
    # att = StateSpaceModels.get_filtered_state(filter)
    # plot(att[:, 1])
    # plot!(att[:, 13])
    # plot(att[:, 2])
    # plot!(att[:, 14])
    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # plot([model.system.y; hcat(forec.expected_value...)'])
    # vline!([192])
end