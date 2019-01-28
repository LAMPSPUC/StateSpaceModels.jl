"""Simulate S future scenarios up to N steps ahead.
    Returns an NxS matrix where each line represents an instant and each column a scenario."""
function simulate(ss::StateSpace, N::Int, S::Int)

    # Number of observations and exogenous variables
    n_exp, p_exp = size(ss.sys.X)

    if p_exp > 0 && N > n_exp - ss.dim.n
        error("Simulation period is larger than period of exogenous variables")
    end

    # Distribution of observation error
    dist_ϵ = MvNormal(zeros(ss.dim.p), ss.param.sqrtH*ss.param.sqrtH')
    # Distribution of state error
    dist_η = MvNormal(zeros(ss.dim.r), ss.param.sqrtQ*ss.param.sqrtQ')

    αsim = Array{Array}(undef, N)
    ysim = Array{Array}(undef, S)

    for s = 1:S
        ysim[s] = Array{Float64}(undef, N, ss.dim.p)

        # Simulating error
        ϵ = rand(dist_ϵ, N)'
        η = rand(dist_η, N)'

        # Generating scenarios from simulated errors
        for t = 1:N
            if t == 1
                αsim[t] = ss.sys.T*ss.state.alpha[ss.dim.n] + (ss.sys.R*η[t, :])
                ysim[s][t, :] = p_exp > 0 ? ss.sys.Z[ss.dim.n + t]*αsim[t] + ϵ[t, :] :
                                            ss.sys.Z[1]*αsim[t] + ϵ[t, :]
            else
                αsim[t] = ss.sys.T*αsim[t-1] + (ss.sys.R*η[t, :])
                ysim[s][t, :] = p_exp > 0 ? ss.sys.Z[ss.dim.n + t]*αsim[t] + ϵ[t, :] :
                                            ss.sys.Z[1]*αsim[t] + ϵ[t, :]
            end
        end
    end

    # Organizing scenarios in matrix form
    scenarios = Array{Float64}(undef, ss.dim.p, N, S)
    for p = 1:ss.dim.p, t = 1:N, s = 1:S
        scenarios[p, t, s] = ysim[s][t, p]
    end

    return scenarios
end
