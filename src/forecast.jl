export forecast, simulate

"""
    forecast(ss::StateSpace{Typ}, N::Int) where Typ

Obtain the minimum mean square error forecasts N steps ahead. Returns the forecasts and the predictive distributions 
at each time period.
"""
function forecast(ss::StateSpace{Typ}, N::Int) where Typ

    # Load estimated covariance matrices
    H = ss.model.H
    Q = ss.model.Q

    # Load system
    n, p, m, r    = size(ss.model)
    Z, T, R, d, c = prepare_forecast(ss, N)

    # Load a, c, P, and F at last in-sample instant
    a0 = ss.smoother.alpha[end, :]
    c0 = ss.model.c[end, :]
    P0 = ss.filter.P[:, :, end]
    F0 = ss.filter.F[:, :, end]
    
    # State and variance forecasts
    a = Matrix{Typ}(undef, N, m)
    P = Array{Typ, 3}(undef, m, m, N)
    F = Array{Typ, 3}(undef, p, p, N)

    # Probability distribution
    dist = Vector{Distribution}(undef, N)

    # Initialization
    a[1, :]    = T*a0 + c0
    P[:, :, 1] = T*P0*T' + R*Q*R'
    F[:, :, 1] = Z[:, :, 1]*P[:, :, 1]*Z[:, :, 1]' + H
    ensure_pos_sym!(F, 1)
    dist[1]    = MvNormal(vec(Z[:, :, 1]*a[1, :] + d[1, :]), F[:, :, 1])

    for t = 2:N
        a[t, :]    = T*a[t-1, :] + c[t-1, :]
        P[:, :, t] = T*P[:, :, t-1]*T' + R*Q*R'
        F[:, :, t] = Z[:, :, t]*P[:, :, t]*Z[:, :, t]' + H
        ensure_pos_sym!(F, t)
        dist[t]    = MvNormal(vec(Z[:, :, t]*a[t, :] + d[t, :]), F[:, :, t])
    end

    forec = Matrix{Typ}(undef, N, p)
    for t = 1:N
        forec[t, :] = mean(dist[t])
    end

    return forec, dist
end

"""
    simulate(ss::StateSpace{Typ}, N::Int, S::Int) where Typ

Simulate S future scenarios up to N steps ahead. Returns a p x N x S matrix where the dimensions represent, respectively,
the number of series in the model, the number of steps ahead, and the number of scenarios.
"""
function simulate(ss::StateSpace{Typ}, N::Int, S::Int) where Typ

    # Load estimated covariance matrices
    H = ss.model.H
    Q = ss.model.Q

    # Load system
    n, p, m, r    = size(ss.model)
    Z, T, R, d, c = prepare_forecast(ss, N)

    # Distribution of the state space errors
    dist_ϵ = MvNormal(H)
    dist_η = MvNormal(Q)

    αsim = Array{Typ, 3}(undef, N, m, S)
    ysim = Array{Typ, 3}(undef, N, p, S)

    for s = 1:S

        # Sampling errors
        ϵ = rand(dist_ϵ, N)'
        η = rand(dist_η, N)'

        # Initializing simulation
        αsim[1, :, s] = T*ss.smoother.alpha[n, :] + c[1, :] + R*η[1, :]
        ysim[1, :, s] = Z[:, :, 1]*αsim[1, :, s] + d[1, :] + ϵ[1, :]

        # Simulating future scenarios
        for t = 2:N
            αsim[t, :, s] = T*αsim[t-1, :, s] + c[t-1, :] + R*η[t, :]
            ysim[t, :, s] = Z[:, :, t]*αsim[t, :, s] + d[t, :] + ϵ[t, :]
        end
    end

    return ysim
end

"""
    prepare_forecast(ss::StateSpace{Typ}, N::Int) where Typ

Adjust matrix Z for forecasting and check for dimension errors.
"""
function prepare_forecast(ss::StateSpace{Typ}, N::Int) where Typ

    # Load system
    n, p, m, r = size(ss.model)
    Z0, T, R   = ztr(ss.model)
    d0 = ss.model.d
    c0 = ss.model.c

    Z = Array{Typ, 3}(undef, p, m, N)
    d = Matrix{Typ}(undef, N, p)
    c = Matrix{Typ}(undef, N, m)
    if ss.model.mode == "time-invariant"
        for t in 1:N, i in 1:p, j in 1:m
            Z[i, j, t] = Z0[i, j, 1]
            d[t, i] = d0[1, i]
            c[t, j] = c0[1, j]
        end
    else
        size(Z0, 3) < n + N && error("Time-variant Z too short for forecasting $N steps ahead")
        size(d0, 1) < n + N && error("Time-variant d too short for forecasting $N steps ahead")
        size(c0, 1) < n + N && error("Time-variant c too short for forecasting $N steps ahead")

        for t in n + 1:n + N, i in 1:p, j in 1:m
            Z[i, j, t - n] = Z0[i, j, t]
            d[t - n, i] = d0[t, i]
            c[t - n, j] = c0[t, j]
        end
    end

    return Z, T, R, d, c
end