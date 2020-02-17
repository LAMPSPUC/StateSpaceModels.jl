using StateSpaceModels, Distributions, LinearAlgebra, Plots, Random

# Fix seed
Random.seed!(1)

# Number of observations
n = 400
# State dimension (2d position + 2d speed)
m = 4
# Measurements dimension (2d noisy position)
p = 2
# Control dimension (2d acceleration)
q = 2

# Damping ratio
ρ = 0.05
# Time delta
Δ = 1.0

# State transition matrix
T = kron(Matrix{Float64}(I, p, p), [1.0 (1.0 - ρ * Δ / 2.0) * Δ; 0.0 (1.0 - ρ * Δ)])
# Input matrix
R = kron(Matrix{Float64}(I, p, p), [0.5 * Δ^2; Δ])
# Output (measurement) matrix
Z = kron(Matrix{Float64}(I, p, p), [1.0 0.0])

# Generate random actuators
Q = 0.5 * Matrix{Float64}(I, q, q)
η = MvNormal(Q)

# Generate random measurement noise
H = 2.0 * Matrix{Float64}(I, p, p)
ε = MvNormal(H)

# Simulate vehicle trajectory
α = zeros(n + 1, m)
y = zeros(n, p)
for t in 1:n
    y[t, :] = Z * α[t, :] + rand(ε)
    α[t + 1, :] = T * α[t, :] + R * rand(η)  
end
α = α[1:n, :]

# User defined model
model = StateSpaceModel(y, Z, T, R)
# Estimate vehicle speed and position
ss = statespace(model)

ss.model.H
ss.model.Q

anim = @animate for i in 1:n
    plot(y[1:i, 1], y[1:i, 2], label="Measured position", line=:scatter, lw=2, markeralpha=0.2, color=:black, title="Vehicle tracking")
    plot!(α[1:i, 1], α[1:i, 3], label="True position", lw=3, color=:indianred)
    plot!(ss.smoother.alpha[1:i, 1], ss.smoother.alpha[1:i, 3], label="Estimated position", lw=2, color=:forestgreen)
end

gif(anim, "vehicle_tracking.gif", fps = 15)