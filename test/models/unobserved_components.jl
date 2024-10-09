@testset "UnobservedComponents" begin
    @test has_fit_methods(UnobservedComponents)

    nile = CSV.File(StateSpaceModels.NILE) |> DataFrame
    model = UnobservedComponents(nile.flow)
    fit!(model)
    @test loglike(model) ≈ -632.5376 atol = 1e-5 rtol = 1e-5
    filt = kalman_filter(model)
    smoother = kalman_smoother(model)
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    finland_fatalities = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
    log_finland_fatalities = log.(finland_fatalities.ff)
    model = UnobservedComponents(log_finland_fatalities; trend = "local linear trend")
    fit!(model)
    @test loglike(model) ≈ 26.740 atol = 1e-5 rtol = 1e-5
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)
    model = UnobservedComponents(log_air_passengers; trend = "local linear trend", seasonal = "stochastic 12")
    fit!(model)
    @test loglike(model) ≈ 234.33641 atol = 1e-5 rtol = 1e-5
    forec = forecast(model, 10)
    @test monotone_forecast_variance(forec)

    rj_temp = CSV.File(StateSpaceModels.RJ_TEMPERATURE) |> DataFrame
    model = UnobservedComponents(rj_temp.Values; trend = "local level", cycle = "stochastic")
    fit!(model)
    # TODO check with other software maybe statsmodels
    @test loglike(model) ≈ -619.8679932 atol = 1e-5 rtol = 1e-5
    filt = kalman_filter(model)
    forec = forecast(model, 52)
    @test monotone_forecast_variance(forec)
    smoother = kalman_smoother(model)
    alpha = get_smoothed_state(smoother)
    @test maximum(alpha[:, 1]) >= 296
    @test minimum(alpha[:, 1]) <= 296
    @test maximum(alpha[:, 2]) >= 7.5
    @test minimum(alpha[:, 2]) <= -7.5

    # Testing that it does not break
    rj_temp = CSV.File(StateSpaceModels.RJ_TEMPERATURE) |> DataFrame
    model = UnobservedComponents(rj_temp.Values; trend = "smooth trend", cycle = "stochastic")
    fit!(model)
end

@testset "UnobservedComponentsExplanatory" begin
    Random.seed!(123)
    @test has_fit_methods(UnobservedComponents)

    nile = CSV.File(StateSpaceModels.NILE) |> DataFrame
    steps_ahead = 10
    nile_y = nile.flow
    X = [0.906 0.247 0.065; 0.443 0.211 0.778; 0.745 0.32 0.081; 0.512 0.011 0.95; 0.253 0.99 0.136; 0.334 0.285 0.345; 0.427 0.078 0.449; 0.867 0.468 0.119; 0.099 0.476 0.04; 0.125 0.157 0.979; 0.692 0.328 0.558; 0.136 0.279 0.91; 0.032 0.022 0.346; 0.35 0.321 0.748; 0.93 0.852 0.115; 0.959 0.023 0.291; 0.774 0.631 0.059; 0.183 0.62 0.563; 0.297 0.434 0.124; 0.15 0.788 0.941; 0.893 0.851 0.298; 0.354 0.623 0.814; 0.131 0.355 0.719; 0.941 0.847 0.823; 0.057 0.719 0.176; 0.245 0.064 0.287; 0.805 0.366 0.027; 0.337 0.28 0.191; 0.823 0.893 0.4; 0.45 0.054 0.463; 0.891 0.661 0.365; 0.711 0.379 0.718; 0.36 0.792 0.0; 0.259 0.398 0.783; 0.39 0.832 0.179; 0.461 0.828 0.061; 0.934 0.575 0.829; 0.753 0.903 0.832; 0.729 0.28 0.71; 0.162 0.505 0.684; 0.637 0.266 0.309; 0.991 0.824 0.73; 0.383 0.371 0.828; 0.618 0.542 0.516; 0.484 0.503 0.101; 0.599 0.443 0.559; 0.453 0.087 0.414; 0.324 0.545 0.931; 0.51 0.596 0.049; 0.656 0.677 0.855; 0.869 0.206 0.392; 0.373 0.78 0.09; 0.692 0.564 0.374; 0.746 0.839 0.649; 0.096 0.342 0.578; 0.459 0.999 0.078; 0.608 0.776 0.99; 0.836 0.719 0.063; 0.283 0.652 0.737; 0.631 0.752 0.883; 0.691 0.081 0.348; 0.821 0.364 0.711; 0.564 0.881 0.045; 0.03 0.6 0.734; 0.091 0.975 0.586; 0.766 0.965 0.567; 0.246 0.14 0.612; 0.096 0.371 0.774; 0.092 0.797 0.055; 0.437 0.096 0.641; 0.231 0.895 0.384; 0.647 0.634 0.325; 0.681 0.612 0.507; 0.122 0.304 0.524; 0.648 0.364 0.023; 0.587 0.294 0.481; 0.377 0.61 0.275; 0.24 0.267 0.63; 0.222 0.24 0.283; 0.509 0.195 0.741; 0.218 0.944 0.804; 0.834 0.72 0.432; 0.563 0.016 0.229; 0.42 0.571 0.976; 0.62 0.31 0.238; 0.292 0.786 0.647; 0.145 0.938 0.018; 0.864 0.918 0.967; 0.926 0.09 0.712; 0.068 0.688 0.407; 0.047 0.739 0.862; 0.426 0.323 0.057; 0.972 0.092 0.697; 0.928 0.761 0.841; 0.077 0.757 0.522; 0.538 0.98 0.388; 0.357 0.843 0.108; 0.471 0.2 0.491; 0.97 0.026 0.486; 0.612 0.999 0.249; 0.539 0.877 0.886; 0.156 0.892 0.617; 0.159 0.57 0.768; 0.485 0.081 0.468; 0.924 0.506 0.68; 0.219 0.068 0.711; 0.457 0.461 0.514; 0.87 0.21 0.412; 0.907 0.305 0.581; 0.688 0.441 0.311]
    β = [ -1.937, 4.389, -4.870]
    X_train = X[1:end - steps_ahead, :]
    X_test  = X[end - steps_ahead + 1:end, :]
    nile_y += X_train*β

    model = UnobservedComponents(nile_y; X = X_train)
    fit!(model)
    @test loglike(model) ≈ -620.58519 atol = 1e-5 rtol = 1e-5
    filt = kalman_filter(model)
    smoother = kalman_smoother(model)
    forec = forecast(model, X_test)
    @test monotone_forecast_variance(forec)

    finland_fatalities = CSV.File(StateSpaceModels.VEHICLE_FATALITIES) |> DataFrame
    log_finland_fatalities = log.(finland_fatalities.ff)
    X = [0.906 0.484 0.926; 0.443 0.599 0.068; 0.745 0.453 0.047; 0.512 0.324 0.426; 0.253 0.51 0.972; 0.334 0.656 0.928; 0.427 0.869 0.077; 0.867 0.373 0.538; 0.099 0.692 0.357; 0.125 0.746 0.471; 0.692 0.096 0.97; 0.136 0.459 0.612; 0.032 0.608 0.539; 0.35 0.836 0.156; 0.93 0.283 0.159; 0.959 0.631 0.485; 0.774 0.691 0.924; 0.183 0.821 0.219; 0.297 0.564 0.457; 0.15 0.03 0.87; 0.893 0.091 0.907; 0.354 0.766 0.688; 0.131 0.246 0.247; 0.941 0.096 0.211; 0.057 0.092 0.32; 0.245 0.437 0.011; 0.805 0.231 0.99; 0.337 0.647 0.285; 0.823 0.681 0.078; 0.45 0.122 0.468; 0.891 0.648 0.476; 0.711 0.587 0.157; 0.36 0.377 0.328; 0.259 0.24 0.279; 0.39 0.222 0.022; 0.461 0.509 0.321; 0.934 0.218 0.852; 0.753 0.834 0.023; 0.729 0.563 0.631; 0.162 0.42 0.62; 0.637 0.62 0.581; 0.991 0.292 0.311; 0.383 0.145 0.121; 0.618 0.864 0.204]
    β = [3.028, 0.086, 0.512]
    X_train = X[1:end - steps_ahead, :]
    X_test  = X[end - steps_ahead + 1:end, :]
    log_finland_fatalities += X_train*β
    model = UnobservedComponents(log_finland_fatalities;X = X_train, trend = "local linear trend")
    fit!(model)
    @test loglike(model) ≈ 21.6918 atol = 1e-5 rtol = 1e-5
    forec = forecast(model, X_test)
    @test monotone_forecast_variance(forec)

    air_passengers = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
    log_air_passengers = log.(air_passengers.passengers)
    X = [0.121 0.575 0.614; 0.137 0.619 0.881; 0.05 0.45 0.098; 0.386 0.515 0.536; 0.846 0.388 0.115; 0.835 0.731 0.818; 0.16 0.04 0.365; 0.904 0.794 0.9; 0.991 0.497 0.354; 0.74 0.396 0.492; 0.229 0.855 0.15; 0.969 0.628 0.544; 0.54 0.964 0.391; 0.546 0.514 0.763; 0.783 0.316 0.799; 0.813 0.152 0.763; 0.83 0.845 0.824; 0.656 0.651 0.374; 0.492 0.655 0.382; 0.1 0.528 0.67; 0.878 0.98 0.573; 0.959 0.956 0.487; 0.634 0.279 0.057; 0.064 0.634 0.291; 0.018 0.443 0.494; 0.16 0.445 0.85; 0.605 0.241 0.269; 0.732 0.819 0.088; 0.439 0.201 0.683; 0.265 0.068 0.738; 0.966 0.321 0.001; 0.783 0.563 0.935; 0.66 0.175 0.698; 0.357 0.975 0.065; 0.455 0.295 0.256; 0.519 0.656 0.04; 0.17 0.448 0.445; 0.072 0.972 0.129; 0.428 0.378 0.492; 0.907 0.067 0.947; 0.991 0.802 0.413; 0.048 0.756 0.986; 0.695 0.641 0.311; 0.319 0.008 0.154; 0.084 0.398 0.522; 0.379 0.21 0.627; 0.298 0.343 0.537; 0.968 0.681 0.117; 0.32 0.469 0.884; 0.315 0.767 0.392; 0.869 0.427 0.456; 0.41 0.392 0.237; 0.204 0.874 0.104; 0.25 0.512 0.376; 0.417 0.078 0.434; 0.806 0.733 0.628; 0.307 0.78 0.457; 0.731 0.628 0.034; 0.553 0.467 0.155; 0.6 0.245 0.304; 0.139 0.825 0.547; 0.052 0.618 0.801; 0.269 0.607 0.652; 0.295 0.861 0.409; 0.602 0.169 0.891; 0.907 0.277 0.555; 0.127 0.504 0.294; 0.865 0.724 0.235; 0.228 0.013 0.193; 0.584 0.698 0.457; 0.157 0.868 0.705; 0.302 0.673 0.641; 0.486 0.359 0.784; 0.915 0.477 0.329; 0.741 0.193 0.023; 0.386 0.936 0.01; 0.92 0.03 0.42; 0.037 0.838 0.788; 0.923 0.745 0.225; 0.199 0.28 0.761; 0.702 0.372 0.745; 0.284 0.81 0.861; 0.301 0.52 0.452; 0.712 0.953 0.624; 0.459 0.574 0.769; 0.097 0.257 0.332; 0.594 0.526 0.014; 0.533 0.906 0.049; 0.94 0.149 0.629; 0.496 0.736 0.392; 0.597 0.313 0.123; 0.129 0.683 0.218; 0.407 0.317 0.398; 0.363 0.509 0.369; 0.487 0.695 0.98; 0.047 0.134 0.24; 0.273 0.727 0.069; 0.457 0.259 0.338; 0.825 0.509 0.401; 0.371 0.092 0.267; 0.186 0.198 0.363; 0.88 0.116 0.889; 0.387 0.041 0.812; 0.535 0.034 0.194; 0.923 0.59 0.38; 0.955 0.073 0.299; 0.765 0.169 0.8; 0.39 0.871 0.788; 0.907 0.919 0.067; 0.46 0.252 0.988; 0.615 0.363 0.861; 0.769 0.098 0.741; 0.046 0.096 0.104; 0.391 0.033 0.121; 0.66 0.142 0.455; 0.589 0.937 0.518; 0.13 0.059 0.895; 0.215 0.644 0.998; 0.938 0.51 0.378; 0.124 0.498 0.954; 0.304 0.966 0.948; 0.411 0.998 0.863; 0.214 0.724 0.539; 0.093 0.907 0.409; 0.33 0.325 0.438; 0.944 0.574 0.093; 0.24 0.36 0.569; 0.533 0.42 0.907; 0.252 0.951 0.448; 0.588 0.154 0.344; 0.019 0.961 0.015; 0.355 0.221 0.353; 0.812 0.145 0.933; 0.118 0.127 0.227; 0.181 0.352 0.502; 0.019 0.45 0.533; 0.757 0.76 0.485; 0.614 0.861 0.739; 0.829 0.892 0.79; 0.602 0.16 0.32; 0.814 0.653 0.382; 0.626 0.185 0.525; 0.133 0.571 0.423; 0.077 0.166 0.477; 0.601 0.862 0.135; 0.243 0.772 0.579; 0.335 0.329 0.651; 0.359 0.462 0.588; 0.229 0.04 0.932; 0.152 0.161 0.84; 0.949 0.695 0.795; 0.92 0.528 0.794; 0.895 0.911 0.64; 0.552 0.543 0.445]
    β = [-0.408, 2.838, 4.718]
    X_train = X[1:end - steps_ahead, :]
    X_test  = X[end - steps_ahead + 1:end, :]
    log_air_passengers += X_train*β
    model = UnobservedComponents(log_air_passengers;X = X_train, trend = "local linear trend", seasonal = "stochastic 12")
    fit!(model)
    @test loglike(model) ≈ 219.978 atol = 1e-5 rtol = 1e-5
    forec = forecast(model, X_test)
    @test monotone_forecast_variance(forec)
    smoother = kalman_smoother(model)
    alpha = get_smoothed_state(smoother)
    @test size(alpha) == (144, 16)

    @test_throws AssertionError simulate_scenarios(model, 10, 1000, ones(5, 2))
    model_sim = deepcopy(model)
    scenarios = simulate_scenarios(model_sim, 10, 100000, X_test)
    test_scenarios_adequacy_with_forecast(forec, scenarios)
    #build X_test as 3D array, where all the scenarios are the same
    X_test_3d = ones(10, 3, 500)
    for i in 1:500
        X_test_3d[:, :, i] = X_test
    end
    model_sim2 = deepcopy(model)
    scenarios = simulate_scenarios(model_sim2, 10, 500, X_test_3d)
    test_scenarios_adequacy_with_forecast(forec, scenarios)

end