"""
    statespace(y::Array{Float64, 1}, s::Int; X = Array{Float64, 2}(undef, 0, 0), nseeds::Int = 3)

Estimate structural model and calculate smoothed and predictive state.
"""
function statespace(y::Array{Float64, 1}, s::Int; X = Array{Float64,2}(undef, 0, 0), nseeds::Int = 3)
    n = length(y)
    y = reshape(y, (n, 1))
    statespace(y, s; X = X, nseeds = nseeds)
end

"""
    statespace(y::Array{Float64, 2}, s::Int; X = Array{Float64, 2}(undef, 0, 0), nseeds::Int = 3)

Estimate time series structural model and calculate smoothed and predictive state.
"""
function statespace(y::Array{Float64, 2}, s::Int; X = Array{Float64, 2}(undef, 0, 0), nseeds::Int = 3)

    # Number of observations and endogenous variables
    n, p = size(y)

    # Number of observations and exogenous variables
    n_exp, p_exp = size(X)

    if p_exp > 0 && n_exp < n
        error("Number of observations in X and y mismatch.")
    end

    @info("Creating structural model with $p endogenous variables and $p_exp exogenous variables.")

    m = (1 + s + p_exp)*p
    r = 3*p

    # Observation equation
    N = max(n, n_exp)
    Z = Array{Array}(undef, N)
    for t = 1:N
        if p_exp > 0
            Z[t] = kron(
                [
                    X[t, :]' 1 0 1 zeros(1, s - 2)
                ],
                Matrix{Float64}(I, p, p)
                )
        else
            Z[t] = kron(
                [
                    1 0 1 zeros(1, s - 2)
                ],
                Matrix{Float64}(I, p, p)
                )
        end
    end
    
    # State equation
    if p_exp > 0
        T0 = [Matrix{Float64}(I, p_exp, p_exp) zeros(p_exp, 1 + s)]
        T = kron(
            [
                T0; 
                zeros(1, p_exp) 1 1 zeros(1, s - 1); 
                zeros(1, p_exp) 0 1 zeros(1, s - 1);
                zeros(1, p_exp) 0 0 -ones(1, s - 1);
                zeros(s - 2, p_exp) zeros(s - 2, 2) Matrix{Float64}(I, s - 2, s - 2) zeros(s - 2)
            ],
            Matrix{Float64}(I, p, p)
            )
    else
        T = kron(
            [
                1 1 zeros(1, s - 1); 
                0 1 zeros(1, s - 1);
                0 0 -ones(1, s - 1);
                zeros(s - 2, 2) Matrix{Float64}(I, s - 2, s - 2) zeros(s - 2)
            ],
            Matrix{Float64}(I, p, p)
            )
    end
    R = kron(
        [
            zeros(p_exp, 3); 
            Matrix{Float64}(I, 3, 3); 
            zeros(s - 2, 3)
        ],
        Matrix{Float64}(I, p, p)
        )
    
    # Creating state space data structures
    dim = StateSpaceDimensions(n, p, m, r, p_exp)
    sys = StateSpaceSystem(y, X, s, Z, T, R)
    
    # Maximum likelihood estimation
    ss_par = estimate_statespace(sys, dim, nseeds)

    # Kalman filter and smoothing
    ss_filter = sqrt_kalmanfilter(sys, dim, ss_par.sqrtH, ss_par.sqrtQ)
    smoothedstate = sqrt_smoother(sys, dim, ss_filter)

    @info("End of structural model estimation.")

    output = StateSpace(sys, dim, smoothedstate, ss_par, ss_filter)

    return output
end
