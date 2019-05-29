export simulate

"""
    simulate(ss::StateSpace, N::Int, S::Int)

Simulate S future scenarios up to N steps ahead. Returns a p x N x S matrix where the dimensions represent, respectively,
the number of series in the model, the number of steps ahead, and the number of scenarios.
"""
function simulate(ss::StateSpace, N::Int, S::Int)

    # Load estimated covariance matrices
    H = ss.param.sqrtH'*ss.param.sqrtH
    Q = ss.param.sqrtQ'*ss.param.sqrtQ

    # Load system matrices
    Z, T, R = ztr(ss.model)
    Z = Z[1]

    # Load a, P, and F at last in-sample instant
    a0 = ss.state.alpha[end]
    P0 = ss.filter.sqrtP[end]'*ss.filter.sqrtP[end]
    F0 = ss.filter.sqrtF[end]'*ss.filter.sqrtF[end]
    
    # State and variance forecasts
    a = Vector{Matrix{Float64}}(undef, N)
    P = Vector{Matrix{Float64}}(undef, N)
    F = Vector{Matrix{Float64}}(undef, N)

    # Probability distribution
    dist = Vector{Distribution}(undef, N)

    # Initialization
    a[1]    = T*a0
    P[1]    = T*P0*T' + R*Q*R'
    F[1]    = Z*P[1]*Z' + H
    dist[1] = MvNormal(vec(Z*a[1]), Symmetric(F[1]))
    sim = Array{Float64}(undef, ss.model.dim.p, N, S)

    for t = 2:N
        a[t] = T*a[t-1]
        P[t] = T*P[t-1]*T' + R*Q*R'
        F[t] = Z*P[t]*Z' + H
        dist[t] = MvNormal(vec(Z*a[t]), Symmetric(F[t]))
    end

    for t = 1:N
        sim[:, t, :] = rand(dist[t], S)
    end

    return sim
end