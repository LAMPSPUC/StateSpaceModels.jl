"""Simulate S future scenarios up to N periods ahead.
    Returns an NxS matrix where each line represents a period and each column represents a scenario."""
function simulate(ss::StateSpace, N::Int, S::Int)

    # Number of observations and exogenous variables
    n_exp, p_exp = size(ss.sys.X)

    if p_exp > 0 && N > n_exp - ss.dim.n
        error("Simulation period is larger than number of observations of future exogenous variables.")
    end

    # Distribution of observation innovation
    dist_ϵ = MvNormal(zeros(ss.dim.p), ss.param.sqrtH*ss.param.sqrtH')
    # Distribution of state innovation
    dist_η = MvNormal(zeros(ss.dim.r), ss.param.sqrtQ*ss.param.sqrtQ')

    αsim = Array{Float64}(N, ss.dim.m)
    ysim = Array{Array}(S)

    for s = 1:S
        ysim[s] = Array{Float64}(N, ss.dim.p)

        # Simulating innovations
        ϵ = rand(dist_ϵ, N)'
        η = rand(dist_η, N)'

        # Generating scenarios from innovations
        for t = 1:N
            if t == 1
                αsim[t, :] = ss.sys.T*ss.state.alpha[ss.dim.n, :] + (ss.sys.R*η[t, :])
                ysim[s][t, :] = ss.sys.Z[t]*αsim[t, :] + ϵ[t, :]
            else
                αsim[t, :] = ss.sys.T*αsim[t-1, :] + (ss.sys.R*η[t, :])
                ysim[s][t, :] = ss.sys.Z[t]*αsim[t, :] + ϵ[t, :]
            end
        end
    end

    # Creating scenarios matrix
    scenarios = Array{Float64}(N, S)
    for t = 1:N, s = 1:S
        scenarios[t, s] = ysim[s][t]
    end

    return scenarios
end
