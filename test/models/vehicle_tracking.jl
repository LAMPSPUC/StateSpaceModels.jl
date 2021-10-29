@testset "Vehicle tracking" begin
    n = 100
    H = [1 0
         0 1.0]
    Q = [1 0
        0 1.0]
    rho = 0.1
    model = VehicleTracking(rand(n, 2), rho, H, Q)

    # Not possible to fit
    @test !has_fit_methods(VehicleTracking)
    
    # Simply test if it runs
    initial_state = [0.0, 0, 0, 0]
    sim = simulate(model.system, initial_state, n)
    
    model = VehicleTracking(sim, 0.1, H, Q)
    kalman_filter(model)
    pos_pred = get_predictive_state(model)
    pos_filtered = get_filtered_state(model)
end