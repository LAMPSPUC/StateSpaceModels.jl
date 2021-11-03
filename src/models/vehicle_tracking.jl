@doc raw"""
    VehicleTracking(y::Matrix{Fl}, ρ::Fl, H::Matrix{Fl}, Q::Matrix{Fl}) where Fl

The vehicle tracking example illustrates a model where there are no hyperparameters, 
the user defines the parameters ``\rho``, ``H`` and ``Q`` and the model gives the predicted and filtered speed and
position. In this case, $y_t$ is a $2 \times 1$ observation vector representing the corrupted measurements 
of the vehicle's position on the two-dimensional plane in instant $t$.

The position and speed in each dimension compose the state of the vehicle. Let us refer to ``x_t^{(d)}`` as 
the position on the axis $d$ and to ``\dot{x}^{(d)}_t`` as the speed on the axis $d$ in instant ``t``. Additionally, 
let ``\eta^{(d)}_t`` be the input drive force on the axis ``d``, which acts as state noise. For a single dimension, 
we can describe the vehicle dynamics as 

```math
\begin{equation}
    \begin{aligned}
        & x_{t+1}^{(d)} = x_t^{(d)} + \Big( 1 - \frac{\rho \Delta_t}{2} \Big) \Delta_t \dot{x}^{(d)}_t + \frac{\Delta^2_t}{2} \eta_t^{(d)}, \\
        & \dot{x}^{(d)}_{t+1} = (1 - \rho) \dot{x}^{(d)}_{t} + \Delta_t \eta^{(d)}_t,
    \end{aligned}\label{eq_control}
\end{equation}
```

We can cast the dynamical system as a state-space model in the following manner:
```math
\begin{align*} 
    y_t &= \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \alpha_{t+1} + \varepsilon_t, \\
    \alpha_{t+1} &= \begin{bmatrix} 1 & (1 - \tfrac{\rho \Delta_t}{2}) \Delta_t & 0 & 0 \\ 0 & (1 - \rho) & 0 & 0 \\ 0 & 0 & 1 & (1 - \tfrac{\rho \Delta_t}{2}) \\ 0 & 0 & 0 & (1 - \rho) \end{bmatrix} \alpha_{t} + \begin{bmatrix} \tfrac{\Delta^2_t}{2} & 0 \\ \Delta_t & 0 \\ 0 & \tfrac{\Delta^2_t}{2} \\ 0 & \Delta_t \end{bmatrix} \eta_{t},
\end{align*}
```

See more on [Vehicle tracking](@ref)
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