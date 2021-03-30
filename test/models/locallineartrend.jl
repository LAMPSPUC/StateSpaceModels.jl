@testset "Local Linear Trend Model" begin
    finland_fatalities = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
    log_finland_fatalities = log.(finland_fatalities.ff)

    @test has_fit_methods(LocalLinearTrend)
    # Based on
    # https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_local_linear_trend.html?highlight=linear%20trend
    # Fitting the model with mod = sm.tsa.UnobservedComponents(df['lff'], 'local linear trend') gives 26.740
    model = LocalLinearTrend(log_finland_fatalities)
    fit!(model)
    @test loglike(model) ≈ 26.740 atol = 1e-5 rtol = 1e-5

    # forecasting
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    predicted_mean = [
        5.933257,
        5.897660,
        5.862062,
        5.826465,
        5.790867,
        5.755270,
        5.719672,
        5.684074,
        5.648477,
        5.612879
    ]
    @test maximum(abs.(predicted_mean .- vcat(forec.expected_value...))) < 1.5e-3
end
