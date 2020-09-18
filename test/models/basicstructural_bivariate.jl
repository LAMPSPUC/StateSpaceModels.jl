@testset "Bivariate Basic Structural Model" begin
    front_rear_df = CSV.read(StateSpaceModels.FRONT_REAR_SEAT_KSI, DataFrame)
    front_rear = [front_rear_df.front_seat_KSI front_rear_df.rear_seats_KSI]

    @test has_fit_methods(StateSpaceModels.BivariateBasicStructural)
  
    model = BivariateBasicStructural(front_rear, 12)
    # This model estimaation is very unstable
    # if it does not converge in 1 over 5 tries with the same 
    # initial vaalues then it is a problem
    for _ in 1:5
        fit!(model)
        if isfitted(model)
            break
        end
    end
    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
    # simualting
    #  TODO add simulation method
    # scenarios = StateSpaceModels.simulate_scenarios(model, 10, 10_000)
    # test_scenarios_adequacy_with_forecast(forec, scenarios)
end