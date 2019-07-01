export simulate

"""
    simulate(ss::StateSpace, N::Int, S::Int)

Simulate S future scenarios up to N steps ahead. Returns a p x N x S matrix where the dimensions represent, respectively,
the number of series in the model, the number of steps ahead, and the number of scenarios.
"""
function simulate(ss::StateSpace, N::Int, S::Int)

    # Load estimated covariance matrices
    H = ss.covariance.H
    Q = ss.covariance.Q

    # Load system matrices
    n, p, m, r = size(ss.model)
    Z0, T, R = ztr(ss.model)

    Z = Array{Float64, 3}(undef, p, m, N)
    if ss.model.mode == "time-invariant"
        Z[:, :, 1:N] .= Z0[:, :, 1]
    else
        size(Z0, 3) < n+N && error("Time-variant Z too short for simulating $N steps ahead")
        Z[:, :, 1:N] .= Z0[:, :, n+1:n+N]
    end

    # Load a, P, and F at last in-sample instant
    a0 = ss.smoother.alpha[end, :]
    P0 = ss.filter.P[:, :, end]
    F0 = ss.filter.F[:, :, end]
    
    # State and variance forecasts
    a = Matrix{Float64}(undef, N, m)
    P = Array{Float64, 3}(undef, m, m, N)
    F = Array{Float64, 3}(undef, p, p, N)

    # Probability distribution
    dist = Vector{Distribution}(undef, N)

    # Initialization
    a[1, :]    = T*a0
    P[:, :, 1]    = T*P0*T' + R*Q*R'
    F[:, :, 1]    = Z[:, :, 1]*P[:, :, 1]*Z[:, :, 1]' + H
    dist[1] = MvNormal(vec(Z[:, :, 1]*a[1, :]), Symmetric(F[:, :, 1]))
    sim = Array{Float64, 3}(undef, ss.model.dim.p, N, S)

    for t = 2:N
        a[t, :] = T*a[t-1, :]
        P[:, :, t] = T*P[:, :, t-1]*T' + R*Q*R'
        F[:, :, t] = Z[:, :, t]*P[:, :, t]*Z[:, :, t]' + H
        dist[t] = MvNormal(vec(Z[:, :, t]*a[t, :]), Symmetric(F[:, :, t]))
    end

    for t = 1:N
        sim[:, t, :] = rand(dist[t], S)
    end

    return sim
end