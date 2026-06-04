@testset "Multivariate Basic Structural Model" begin
    # Multivariate
    front_rear_df = CSV.File(StateSpaceModels.FRONT_REAR_SEAT_KSI) |> DataFrame
    front_rear = [front_rear_df.front_seat_KSI front_rear_df.rear_seats_KSI]

    model = MultivariateBasicStructural(front_rear, 12)
    fit!(model)
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simualting
    scenarios = simulate_scenarios(model, 10, 10_000)
    test_scenarios_adequacy_with_forecast(forec, scenarios)

    # Regression test for issue #340
    n, p = size(front_rear)
    m = num_states(model)

    filter = kalman_filter(model)
    smoother = kalman_smoother(model)

    # Innovations variance F - shape (n, p, p)
    @test size(get_innovations_variance(filter)) == (n, p, p)
    # Filtered-state variance Ptt - shape (n, m, m)
    @test size(get_filtered_state_variance(filter)) == (n, m, m)
    # Predictive-state variance P - shape (n+1, m, m)
    @test size(get_predictive_state_variance(filter)) == (n + 1, m, m)
    # Smoothed-state variance  - shape (n, m, m)
    @test size(get_smoothed_state_variance(smoother)) == (n, m, m)
end