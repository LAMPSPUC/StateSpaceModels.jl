@testset "DAR Model" begin
    # Univariate
    sunspot_year = CSV.File(StateSpaceModels.SUNSPOTS_YEAR) |> DataFrame

    @test has_fit_methods(DAR)

    model = DAR(sunspot_year.value, 9)
    fit!(model)
    # Runned on Python statsmodels
    @test loglike(model) ≈ -1175.9129 atol = 2.0 rtol = 1e-3

    @test_throws ErrorException DAR(vcat(rand(10), NaN, rand(10)), 2)

    # forecasting
    @test_throws ErrorException forecast(model, 10)
end