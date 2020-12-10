@doc raw"""
TODO
"""
mutable struct VehicleTracking <: StateSpaceModel
    system::LinearMultivariateTimeInvariant

    function VehicleTracking(y::Matrix{Fl}, ρ::Fl, H::Matrix{Fl}, Q::Matrix{Fl}) where Fl
        p = 2
        Z = kron(Matrix{Fl}(I, p, p), [1.0 0.0])
        T = kron(Matrix{Fl}(I, p, p), [1 (1 - ρ / 2); 0 (1 - ρ)])
        R = kron(Matrix{Fl}(I, p, p), [0.5; 1])
        d = zeros(Fl, p)
        c = zeros(Fl, 4)
        H = H
        Q = Q

        system = LinearMultivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        return new(system)
    end
end

function default_filter(model::VehicleTracking)
    Fl = typeof_model_elements(model)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, num_states(model))
    skip_llk_instants = length(a1)
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    return MultivariateKalmanFilter(2, a1, P1, skip_llk_instants, steadystate_tol)
end
# H = [1 0
#      0 1.0]
# Q = [1 0
#     0 1.0]
# rho = 0.1
# model = VehicleTracking(rand(100, 2), rho, H, Q)

# n = 100
# initial_state = [0.0, 0, 0, 0]
# sim = simulate(model.system, initial_state, 100)

# model = VehicleTracking(sim, 0.1, H, Q)
# kalman_filter(model)
# pos_filter = get_filtered_state(model)
# pos_smoother = get_smoothed_state(model)

# using Plots
# scatter(sim[:, 1], sim[:, 2])
# plot!(pos_filter[:, 1], pos_filter[:, 3])
# plot!(pos_smoother[:, 1], pos_smoother[:, 3])

