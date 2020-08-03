@testset "ARIMA" begin
    internet = read_csv(StateSpaceModels.INTERNET)
    dinternet = internet.dinternet[2:end]
    @test has_fit_methods(ARIMA)

    model = ARIMA(dinternet; order=(1, 0, 1))
    fit(model)
    @test loglike(model) ≈ -254.149 atol = 1e-5 rtol = 1e-5

    missing_obs = [6, 16, 26, 36, 46, 56, 66, 72, 73, 74, 75, 76, 86, 96]
    missing_dinternet = copy(dinternet)
    missing_dinternet[missing_obs] .= NaN

    model = ARIMA(missing_dinternet; order=(1, 0, 1))
    fit(model)
    @test_broken loglike(model) ≈ -225.770 atol = 1e-5 rtol = 1e-5

    wpi = read_csv(StateSpaceModels.WHOLESALE_PRICE_INDEX).wpi
    model = ARIMA(wpi; order=(1, 1, 1))
    fit(model)
    @test loglike(model) ≈ -137.246818 atol = 1e-5 rtol = 1e-5
end