"""
    build_statespace(model::BasicStructuralModel)

Build state-space system for a given structural model.
"""
function build_statespace(model::BasicStructuralModel)

    # Number of endogenous observations and variables
    n, p = size(model.y)

    @info("Building basic structural model with $p endogenous variables.")

    s = model.s
    m = (1 + s)*p
    r = 3*p

    # Observation equation
    Z = Vector{Array}(undef, n)
    for t = 1:n
        Z[t] = kron(
            [
                1 0 1 zeros(1, s - 2)
            ],
            Matrix{Float64}(I, p, p)
            )
    end
    
    # State equation
    T = kron(
        [
            1 1 zeros(1, s - 1); 
            0 1 zeros(1, s - 1);
            0 0 -ones(1, s - 1);
            zeros(s - 2, 2) Matrix{Float64}(I, s - 2, s - 2) zeros(s - 2)
        ],
        Matrix{Float64}(I, p, p)
        )

    R = kron(
        [
            Matrix{Float64}(I, 3, 3); 
            zeros(s - 2, 3)
        ],
        Matrix{Float64}(I, p, p)
        )

    dim = StateSpaceDimensions(n, p, m, r)
    sys = StateSpaceSystem(model.y, Z, T, R, dim)

    return sys

end

"""
    build_statespace(model::StructuralModelExogenous)

Build state-space system for a given structural model.
"""
function build_statespace(model::StructuralModelExogenous)

    # Number of endogenous observations and variables
    n, p = size(model.y)

    # Number of exogenous observations and variables
    n_exp, p_exp = size(model.X)

    if p_exp > 0 && n_exp < n
        error("Number of observations in X must be greater than or equal to the number of observations in y.")
    end

    @info("Building structural model with $p endogenous variables and $p_exp exogenous variables.")

    s = model.s
    m = (1 + s + p_exp)*p
    r = 3*p

    # Observation equation
    N = max(n, n_exp)
    Z = Vector{Array}(undef, N)
    for t = 1:N
        Z[t] = kron(
            [
                model.X[t, :]' 1 0 1 zeros(1, s - 2)
            ],
            Matrix{Float64}(I, p, p)
            )
    end
    
    # State equation
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

    R = kron(
        [
            zeros(p_exp, 3); 
            Matrix{Float64}(I, 3, 3); 
            zeros(s - 2, 3)
        ],
        Matrix{Float64}(I, p, p)
        )

    dim = StateSpaceDimensions(n, p, m, r)
    sys = StateSpaceSystem(model.y, Z, T, R, dim)

    return sys

end