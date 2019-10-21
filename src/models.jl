export structural, local_level, linear_trend

"""
    structural(y::VecOrMat{Typ}, s::Int; X::VecOrMat{Typ} = Matrix{Typ}(undef, 0, 0)) where Typ <: Real

Build state-space system for a given structural model with observations `y`, seasonality `s`, and, optionally, exogenous variables `X`.

If `y` is provided as an `Array{Typ, 1}` it will be converted to an `Array{Typ, 2}` inside the `StateSpaceModel`. The same will happen to X, 
if an `Array{Typ, 1}` it will be converted to an `Array{Typ, 2}` inside the `StateSpaceModel`.
"""
function structural(y::VecOrMat{Typ}, s::Int; X::VecOrMat{Typ} = Matrix{Typ}(undef, 0, 0)) where Typ <: Real

    # Number of observations and endogenous variables
    y = y[:, :]
    n, p = size(y)

    # Number of observations and exogenous variables
    X = X[:, :]
    n_exp, p_exp = size(X)

    if p_exp > 0 && n_exp < n
        error("Number of observations in X must be greater than or equal to the number of observations in y.")
    end

    m = (1 + s + p_exp)*p
    r = 3*p

    # Observation equation
    N = max(n, n_exp)
    if p_exp > 0 # exogenous variables: Z is time-variant
        Z = Array{Typ, 3}(undef, p, m, N)
        for t = 1:N
            Z[:, :, t] = kron(
                Matrix{Typ}(I, p, p),
                [
                    X[t, :]' 1 0 1 zeros(1, s - 2)
                ]
                )
        end
    else # no exogenous variables: Z is time-invariant
        Z = kron(
            Matrix{Typ}(I, p, p),
            [
                1 0 1 zeros(1, s - 2)
            ]
            )
    end
    
    # State equation
    if p_exp > 0
        T0 = [Matrix{Typ}(I, p_exp, p_exp) zeros(p_exp, 1 + s)]
        T = kron(
            Matrix{Typ}(I, p, p),
            [
                T0; 
                zeros(1, p_exp) 1 1 zeros(1, s - 1); 
                zeros(1, p_exp) 0 1 zeros(1, s - 1);
                zeros(1, p_exp) 0 0 -ones(1, s - 1);
                zeros(s - 2, p_exp) zeros(s - 2, 2) Matrix{Typ}(I, s - 2, s - 2) zeros(s - 2)
            ]
            )
    else
        T = kron(
            Matrix{Typ}(I, p, p),
            [
                1 1 zeros(1, s - 1); 
                0 1 zeros(1, s - 1);
                0 0 -ones(1, s - 1);
                zeros(s - 2, 2) Matrix{Typ}(I, s - 2, s - 2) zeros(s - 2)
            ]
            )
        end

    R = kron(
        Matrix{Typ}(I, p, p),
        [
            zeros(p_exp, 3); 
            Matrix{Typ}(I, 3, 3); 
            zeros(s - 2, 3)
        ]
        )

    return StateSpaceModel(y, Z, T, R)

end

"""
    local_level(y::VecOrMat{Typ}) where Typ <: Real

Build state-space system for a local level model with observations y.

If `y` is provided as an `Array{Typ, 1}` it will be converted to an `Array{Typ, 2}` inside the `StateSpaceModel`.
"""
function local_level(y::VecOrMat{Typ}) where Typ <: Real

    # Number of observations and endogenous variables
    y = y[:, :]
    n, p = size(y)

    m = p
    r = p

    # Observation equation
    Z = Matrix{Typ}(I, p, p)

    # State equation
    T = Matrix{Typ}(I, p, p)
    R = Matrix{Typ}(I, p, p)

    return StateSpaceModel(y, Z, T, R)
end

"""
    linear_trend(y::VecOrMat{Typ}) where Typ <: Real

Build state-space system for a linear trend model with observations y.

If `y` is provided as an `Array{Typ, 1}` it will be converted to an `Array{Typ, 2}` inside the `StateSpaceModel`.
"""
function linear_trend(y::VecOrMat{Typ}) where Typ <: Real

    # Number of observations and endogenous variables
    y = y[:, :]
    n, p = size(y)

    m = 2*p
    r = 2*p

    # Observation equation
    Z = kron(Matrix{Typ}(I, p, p),[1 0])

    # State equation
    T = kron(Matrix{Typ}(I, p, p),[1 1; 0 1])
    R = kron(Matrix{Typ}(I, p, p),[1 0; 0 1])

    return StateSpaceModel(y, Z, T, R)
end
