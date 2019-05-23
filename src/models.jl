"""
    structuralmodel(y::VecOrMat{Typ}, s::Int; X::VecOrMat{Typ} = Matrix{Float64}(undef, 0, 0)) where Typ <: AbstractFloat

Build state-space system for a given structural model with observations y, seasonality s, and, optionally, exogenous variables X.
"""
function structuralmodel(y::VecOrMat{Typ}, s::Int; X::VecOrMat{Typ} = Matrix{Float64}(undef, 0, 0)) where Typ <: AbstractFloat

    # Number of observations and endogenous variables
    n, p = size(y[:, :])

    # Number of observations and exogenous variables
    n_exp, p_exp = size(X[:, :])

    if p_exp > 0 && n_exp < n
        error("Number of observations in X must be greater than or equal to the number of observations in y.")
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

    dim = StateSpaceDimensions(n, p, m, r)
    model = StateSpaceModel(y, Z, T, R, dim)

    return model

end