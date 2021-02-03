using StateSpaceModels, LinearAlgebra

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

function StateSpaceModels.default_filter(model::VehicleTracking)
    Fl = typeof_model_elements(model)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, num_states(model))
    skip_llk_instants = length(a1)
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    return MultivariateKalmanFilter(2, a1, P1, skip_llk_instants, steadystate_tol)
end
n = 400
H = [1 0
     0 1.0]
Q = [1 0
    0 1.0]
rho = 0.1
model = VehicleTracking(rand(n, 2), rho, H, Q)

initial_state = [0.0, 0, 0, 0]
sim = simulate(model.system, initial_state, n)

model = VehicleTracking(sim, 0.1, H, Q)
kalman_filter(model)
pos_pred = get_predictive_state(model)
pos_filtered = get_filtered_state(model)

using Plots
anim = @animate for i in 1:n
    plot(sim[1:i, 1], sim[1:i, 2], label="Measured position", line=:scatter, lw=2, markeralpha=0.2, color=:black, title="Vehicle tracking")
    plot!(pos_pred[1:i+1, 1], pos_pred[1:i+1, 3], label = "Predicted position", lw=2, color=:forestgreen)
    plot!(pos_filtered[1:i, 1], pos_filtered[1:i, 3], label = "Filtered position", lw=2, color=:indianred)
end
gif(anim, "anim_fps15.gif", fps = 15)