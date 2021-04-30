@testset "Local Level With Explanatory Model" begin
    @test has_fit_methods(LocalLevelExplanatory)
    y = [
    0.8297606696482265 
    0.5151262659173474 
    0.6578708499069135 
    0.15479984562134974
    0.41836893137914033
    0.46381576757801835
    0.18373491281112986
    0.09712212189897196
    0.9696168042329114 
    0.8623172534114631 
    0.4312662075760265 
    0.17938380947382027
    0.6265684534572271 
    0.6395165239895426 
    0.370025441453405
    ]
    X = [
    0.221409   0.646901
    0.943735   0.163734
    0.19864    0.620721
    0.462932   0.876136
    0.88467    0.673716
    0.387496   0.415591
    0.29221    0.989594
    0.487926   0.721127
    0.0321253  0.318292
    0.742561   0.582234
    0.664695   0.749821
    0.791058   0.423212
    0.0652515  0.955714
    0.842513   0.788755
    0.198105   0.79092 
    ]
    model = LocalLevelExplanatory(y, X)
    fit!(model)

    # forecasting
    # For a fixed forecasting explanatory the variance must not decrease
    forec = forecast(model, ones(5, 2))
    @test monotone_forecast_variance(forec)
    kf = kalman_filter(model)
    ks = kalman_smoother(model)
end
