export structuralmodel, locallevelmodel, lineartrendmodel

"""
    structuralmodel(y::VecOrMat{Typ}, s::Int; X::VecOrMat{Typ} = Matrix{Float64}(undef, 0, 0)) where Typ <: AbstractFloat

Build state-space system for a given structural model with observations y, seasonality s, and, optionally, exogenous variables X.

If `y` is proided as an `Array{Typ, 1}` it will be converted to an `Array{Typ, 2}` inside the `StateSpaceModel`. The same will happen to X, 
if an `Array{Typ, 1}` it will be converted to an `Array{Typ, 2}` inside the `StateSpaceModel`.
"""
function structuralmodel(y::VecOrMat{Typ}, s::Int; X::VecOrMat{Typ} = Matrix{Float64}(undef, 0, 0)) where Typ <: AbstractFloat

    # Number of observations and endogenous variables
    y = y[:, :]
    n, p = size(y)

    # Number of observations and exogenous variables
    X = X[:, :]
    n_exp, p_exp = size(X)

    if p_exp > 0 && n_exp < n
        error("Number of observations in X must be greater than or equal to the number of observations in y.")
    end

    @info("Creating structural model with $p endogenous variables and $p_exp exogenous variables.")

    m = (1 + s + p_exp)*p
    r = 3*p

    # Observation equation
    N = max(n, n_exp)
    Z = []
    if p_exp > 0 # exogenous variables: Z is time-variant
        Z = Vector{Matrix{Float64}}(undef, N)
        for t = 1:N
            Z[t] = kron(
                Matrix{Float64}(I, p, p),
                [
                    X[t, :]' 1 0 1 zeros(1, s - 2)
                ]
                )
        end
    else # no exogenous variables: Z is time-invariant
        Z = kron(
            Matrix{Float64}(I, p, p),
            [
                1 0 1 zeros(1, s - 2)
            ]
            )
    end
    
    # State equation
    if p_exp > 0
        T0 = [Matrix{Float64}(I, p_exp, p_exp) zeros(p_exp, 1 + s)]
        T = kron(
            Matrix{Float64}(I, p, p),
            [
                T0; 
                zeros(1, p_exp) 1 1 zeros(1, s - 1); 
                zeros(1, p_exp) 0 1 zeros(1, s - 1);
                zeros(1, p_exp) 0 0 -ones(1, s - 1);
                zeros(s - 2, p_exp) zeros(s - 2, 2) Matrix{Float64}(I, s - 2, s - 2) zeros(s - 2)
            ]
            )
    else
        T = kron(
            Matrix{Float64}(I, p, p),
            [
                1 1 zeros(1, s - 1); 
                0 1 zeros(1, s - 1);
                0 0 -ones(1, s - 1);
                zeros(s - 2, 2) Matrix{Float64}(I, s - 2, s - 2) zeros(s - 2)
            ]
            )
        end

    R = kron(
        Matrix{Float64}(I, p, p),
        [
            zeros(p_exp, 3); 
            Matrix{Float64}(I, 3, 3); 
            zeros(s - 2, 3)
        ]
        )

    dim = StateSpaceDimensions(n, p, m, r)
    model = StateSpaceModel(y, Z, T, R, dim, "time-variant")

    return model

end

"""
    locallevelmodel(y::VecOrMat{Typ}) where Typ <: AbstractFloat

Build state-space system for a local level model with observations y.

If `y` is proided as an `Array{Typ, 1}` it will be converted to an `Array{Typ, 2}` inside the `StateSpaceModel`.
"""
function locallevelmodel(y::VecOrMat{Typ}) where Typ <: AbstractFloat

    # Number of observations and endogenous variables
    y = y[:, :]
    n, p = size(y)

    @info("Creating local level model with $p endogenous variables.")

    m = p
    r = p

    # Observation equation
    Z = Matrix{Float64}(I, p, p)

    # State equation
    T = Matrix{Float64}(I, p, p)
    R = Matrix{Float64}(I, p, p)

    dim = StateSpaceDimensions(n, p, m, r)
    model = StateSpaceModel(y, Z, T, R, dim, "time-invariant")

    return model
end

"""
    lineartrendmodel(y::VecOrMat{Typ}) where Typ <: AbstractFloat

Build state-space system for a linear trend model with observations y.

If `y` is proided as an `Array{Typ, 1}` it will be converted to an `Array{Typ, 2}` inside the `StateSpaceModel`.
"""
function lineartrendmodel(y::VecOrMat{Typ}) where Typ <: AbstractFloat

    # Number of observations and endogenous variables
    y = y[:, :]
    n, p = size(y)

    @info("Creating linear trend model with $p endogenous variables.")

    m = 2*p
    r = 2*p

    # Observation equation
    Z = kron(Matrix{Float64}(I, p, p),[1 0])

    # State equation
    T = kron(Matrix{Float64}(I, p, p),[1 1; 0 1])
    R = kron(Matrix{Float64}(I, p, p),[1 0; 0 1])

    dim = StateSpaceDimensions(n, p, m, r)
    model = StateSpaceModel(y, Z, T, R, dim, "time-invariant")

    return model
end