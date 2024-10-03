@testset "UnobservedComponentsExplanatory" begin
    Random.seed!(123)
    @test has_fit_methods(UnobservedComponentsExplanatory)

    nile = CSV.File(StateSpaceModels.NILE) |> DataFrame
    steps_ahead = 10
    nile_y = nile.flow
    X = rand(length(nile_y) + steps_ahead, 3)
    β = rand(Normal(0, 1), 3)*3
    X_train = X[1:end - steps_ahead, :]
    X_test  = X[end - steps_ahead + 1:end, :]
    nile_y += X_train*β

    model = UnobservedComponentsExplanatory(nile_y, X_train)
    fit!(model)
    @test loglike(model) ≈ -620.58519 atol = 1e-5 rtol = 1e-5
    filt = kalman_filter(model)
    smoother = kalman_smoother(model)
    forec = forecast(model, X_test)
    @test monotone_forecast_variance(forec)

    finland_fatalities = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
    log_finland_fatalities = log.(finland_fatalities.ff)
    X = rand(length(log_finland_fatalities) + steps_ahead, 3)
    β = rand(Normal(0, 1), 3)*3
    X_train = X[1:end - steps_ahead, :]
    X_test  = X[end - steps_ahead + 1:end, :]
    log_finland_fatalities += X_train*β
    model = UnobservedComponentsExplanatory(log_finland_fatalities, X_train; trend = "local linear trend")
    fit!(model)
    @test loglike(model) ≈ 20.12409568926452 atol = 1e-5 rtol = 1e-5
    forec = forecast(model, X_test)
    @test monotone_forecast_variance(forec)

    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)
    X = rand(length(log_air_passengers) + steps_ahead, 3)
    β = rand(Normal(0, 1), 3)*3
    X_train = X[1:end - steps_ahead, :]
    X_test  = X[end - steps_ahead + 1:end, :]
    log_air_passengers += X_train*β
    model = UnobservedComponentsExplanatory(log_air_passengers, X_train; trend = "local linear trend", seasonal = "stochastic 12")
    fit!(model)
    @test loglike(model) ≈ 223.82229 atol = 1e-5 rtol = 1e-5
    forec = forecast(model, X_test)
    @test monotone_forecast_variance(forec)
    smoother = kalman_smoother(model)
    alpha = get_smoothed_state(smoother)
    @test size(alpha) == (144, 16)

    # rj_temp = CSV.File(RJ_TEMPERATURE) |> DataFrame
    # rj_tem_values = rj_temp.Values ./ 100
    # steps_ahead = 52
    # X = rand(length(rj_tem_values) + steps_ahead, 1)
    # β = [0.1]
    # X_train = X[1:end - steps_ahead, :]
    # X_test  = X[end - steps_ahead + 1:end, :]
    # rj_tem_values += X_train*β
    # model = UnobservedComponentsExplanatory(rj_tem_values, X_train; trend = "local level", cycle = "stochastic")
    # fit!(model)
    # model.hyperparameters
    # forec = forecast(model, X_test)

    # TODO check with other software maybe statsmodels
    # @test loglike(model) ≈ -1032.40953 atol = 1e-5 rtol = 1e-5
    # filt = kalman_filter(model)
    # forec = forecast(model, X_test)
    # @test monotone_forecast_variance(forec)
    # smoother = kalman_smoother(model)
    # alpha = get_smoothed_state(smoother)
    # @test maximum(alpha[:, 1]) >= 296
    # @test minimum(alpha[:, 1]) <= 296
    # @test maximum(alpha[:, 2]) >= 7.5
    # @test minimum(alpha[:, 2]) <= -7.5

    # # Testing that it does not break
    # rj_temp = CSV.File(RJ_TEMPERATURE) |> DataFrame
    # model = UnobservedComponents(rj_temp.Values; trend = "smooth trend", cycle = "stochastic")
    # fit!(model)
end