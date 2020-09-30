@testset "Multivariate Basic Structural Model" begin
    # Multivariate
    front_rear_df = CSV.read(StateSpaceModels.FRONT_REAR_SEAT_KSI, DataFrame)
    front_rear = [front_rear_df.front_seat_KSI front_rear_df.rear_seats_KSI]
  
    model = MultivariateBasicStructural(front_rear, 12)
    fit!(model)
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)
end