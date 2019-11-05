export structural, local_level, linear_trend, regression

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

"""
    regression(y::VecOrMat{Typ}, X::VecOrMat{Typ}) where Typ <: Real

Build state-space system for estimating a regression model ``y_t = X_t\\beta_t + \\varpsilon_t``. Once the model is estimated
the user can recover the parameter ``\\hat \\beta`` by the querying the smoothed states of the model.
"""
function regression(y::VecOrMat{Typ}, X::VecOrMat{Typ}) where Typ <: Real
    # Certify that they are matrices
    y = ensure_is_matrix(y)
    X = ensure_is_matrix(X)
    # Assert sizes are adequate
    n_y, p_y = size(y)
    n_X, p_X = size(X)
    @assert n_X == n_y

    if p_y > 1
        error("StateSpaceModels currently supports regression only for univariate cases.")
    end

    # Fill Z
    Z = Array{Typ, 3}(undef, p_y, p_X, n_X)
    for t in 1:n_X
        Z[:, :, t] = X[t, :]
    end
    T = Matrix{Typ}(I, p_X, p_X)
    R = zeros(Typ, p_X, 1)
    Q = zeros(Typ, 1, 1)
    d = zeros(n_y, p_y)
    c = zeros(n_X, p_X)
    # H is the only variable to estimate
    H = build_H(p_y, Typ)

    return StateSpaceModel(y, Z, T, R, d, c, H, Q)
end
