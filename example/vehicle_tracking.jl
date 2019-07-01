
using Distributions, LinearAlgebra, Random

# Number of observations
n = 100
# State dimension (2d position + 2d speed)
m = 4
# Measurements dimension (2d noisy position)
p = 2
# Control dimension (2d acceleration)
q = 2

# Damping ratio
damping = .05
# Time delta
Δ = 1.

# State transition matrix
T = kron(Matrix{Float64}(I, p, p), [1. (1. - damping * Δ / 2.) * Δ; 0. (1. - damping* Δ)])
# Input matrix
R = kron(Matrix{Float64}(I, p, p), [.5 * Δ^2.; Δ])
# Output (measurement) matrix
Z = kron(Matrix{Float64}(I, p, p), [1. 0])

# Generate random actuators
Q = 1.5 * Matrix{Float64}(I, q, q)
η = MvNormal(zeros(q), Q)

# Generate random measurement noise
H = .1 * Matrix{Float64}(I, p, p)
ε = MvNormal(zeros(p), H)

# Simulate vehicle trajectory
α = zeros(m, n + 1)
y = zeros(p, n)

for t in 1:n
    y[:, t] = Z * α[:, t] + rand(ε)
    α[:, t + 1] = T * α[:, t] + R * rand(η)  
end


