export StateSpaceDimensions, StateSpaceModel, SmoothedState,
        FilterOutput, StateSpace

export KalmanFilter, SquareRootFilter, UnivariateKalmanFilter

# Abstract types
"""
    AbstractFilter

Abstract type used to implement an interface for generic filters.
"""
abstract type AbstractFilter end

"""
    AbstractSmoother

Abstract type used to implement an interface for generic smoothers.
"""
abstract type AbstractSmoother end

"""
    AbstractOptimizationMethod{T}

Abstract type used to implement an interface for generic optimization methods.
"""
abstract type AbstractOptimizationMethod{T} end

# Auxiliary structure for square-root Kalman filter
mutable struct SquareRootFilter{T <: Real} <: AbstractFilter
    a::Matrix{T} # predictive state
    v::Matrix{T} # innovations
    sqrtP::Array{T, 3} # lower triangular matrix with sqrt-covariance of the predictive state
    sqrtF::Array{T, 3} # lower triangular matrix with sqrt-covariance of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
    K::Array{T, 3} # Kalman gain
end

# Auxiliary structure for Kalman filter
mutable struct KalmanFilter{T <: Real} <: AbstractFilter
    a::Matrix{T} # predictive state
    att::Matrix{T}
    v::Matrix{T} # innovations
    P::Array{T, 3} # covariance matrix of the predictive state
    Ptt::Array{T, 3}
    F::Array{T, 3} # covariance matrix of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
    K::Array{T, 3} # Kalman gain
end

# Auxiliary structure for univariate Kalman filter
mutable struct UnivariateKalmanFilter{T <: Real} <: AbstractFilter
    a::Matrix{T} # predictive state
    att::Matrix{T}
    v::Vector{T} # innovations
    P::Array{T, 3} # covariance matrix of the predictive state
    Ptt::Array{T, 3}
    F::Vector{T} # covariance matrix of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
    K::Matrix{T} # Kalman gain
end

# Auxiliary structure for smoother
mutable struct Smoother{T <: Real} <: AbstractSmoother
    alpha::Matrix{T} # smoothed state
    V::Array{T, 3} # variance of smoothed state
end

"""
    StateSpaceDimensions

StateSpaceModel dimensions, following the notation of on the book
\"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `n` is the number of observations
* `p` is the dimension of the observation vector ``y_t``
* `m` is the dimension of the state vector ``\\alpha_t``
* `r` is the dimension of the state covariance matrix ``Q``
"""
struct StateSpaceDimensions
    n::Int
    p::Int
    m::Int
    r::Int
end

"""
    StateSpaceModel{Typ <: Real}

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `y` A ``n \\times p`` matrix containing observations
* `Z` A ``p \\times m \\times n`` matrix
* `T` A ``m \\times m`` matrix
* `R` A ``m \\times r`` matrix
* `d` A ``n \\times p`` matrix
* `c` A ``n \\times m`` matrix
* `H` A ``p \\times p`` matrix
* `Q` A ``r \\times r`` matrix
"""
struct StateSpaceModel{Typ <: Real}
    y::Matrix{Typ} # Observations
    Z::Array{Typ, 3} # Observation matrix
    T::Matrix{Typ} # State matrix
    R::Matrix{Typ} # State error matrix
    d::Matrix{Typ} # Mean adjustments for the observation equation
    c::Matrix{Typ} # Mean adjustments for the state equation
    H::Matrix{Typ} # Covariance matrix of the observation vector
    Q::Matrix{Typ} # Covariance matrix of the state vector
    dim::StateSpaceDimensions
    missing_observations::Vector{Int}
    mode::String

    function StateSpaceModel(y::VecOrMat{Typ}, Z::Array{Typ, 3}, T::Matrix{Typ}, R::Matrix{Typ},
                             d::Matrix{Typ}, c::Matrix{Typ}, H::Matrix{Typ}, Q::Matrix{Typ}) where Typ <: Real

        # Convert y to a Matrix
        y = ensure_is_matrix(y)

        # Build StateSpaceDimensions
        dim = build_ss_dim(y, Z, T, R)

        # Ensure dimensions are coherent
        assert_dimensions(c, d, dim)

        return new{Typ}(y, Z, T, R, d, c, H, Q, dim, find_missing_observations(y), "time-variant")
    end

    function StateSpaceModel(y::VecOrMat{Typ}, Z::Matrix{Typ}, T::Matrix{Typ}, R::Matrix{Typ},
                             d::Matrix{Typ}, c::Matrix{Typ}, H::Matrix{Typ}, Q::Matrix{Typ}) where Typ <: Real

        # Convert y to a Matrix
        y = ensure_is_matrix(y)

        # Build StateSpaceDimensions
        dim = build_ss_dim(y, Z, T, R)

        Zvar = Array{Typ, 3}(undef, dim.p, dim.m, dim.n)
        for t in 1:dim.n, i in axes(Z, 1), j in axes(Z, 2)
            Zvar[i, j, t] = Z[i, j]
        end

        # Ensure dimensions are coherent
        assert_dimensions(c, d, dim)

        return new{Typ}(y, Zvar, T, R, d, c, H, Q, dim, find_missing_observations(y), "time-invariant")
    end

    function StateSpaceModel(y::VecOrMat{Typ}, Z::Array{Typ, 3}, T::Matrix{Typ}, R::Matrix{Typ}) where Typ <: Real

        # Convert y to a Matrix
        y = ensure_is_matrix(y)

        # Build StateSpaceDimensions
        dim = build_ss_dim(y, Z, T, R)

        # Build c and d with zeros
        d = zeros(Typ, size(Z, 3), dim.p)
        c = zeros(Typ, size(Z, 3), dim.m)

        # Build H and Q matrices with NaNs
        H = build_H(dim.p, Typ)
        Q = build_Q(dim.r, dim.p, Typ)

        # Ensure dimensions are coherent
        assert_dimensions(c, d, dim)

        return new{Typ}(y, Z, T, R, d, c, H, Q, dim, find_missing_observations(y), "time-variant")
    end

    function StateSpaceModel(y::VecOrMat{Typ}, Z::Matrix{Typ}, T::Matrix{Typ}, R::Matrix{Typ}) where Typ <: Real

        # Convert y to a Matrix
        y = ensure_is_matrix(y)

        # Build StateSpaceDimensions
        dim = build_ss_dim(y, Z, T, R)

        # Build Z
        Zvar = Array{Typ, 3}(undef, dim.p, dim.m, dim.n)
        for t in 1:dim.n, i in axes(Z, 1), j in axes(Z, 2)
            Zvar[i, j, t] = Z[i, j] # Zvar[:, :, t] = Z
        end

        # Build c and d with zeros
        d = zeros(Typ, dim.n, dim.p)
        c = zeros(Typ, dim.n, dim.m)

        # Build H and Q matrices with NaNs
        H = build_H(dim.p, Typ)
        Q = build_Q(dim.r, dim.p, Typ)

        # Ensure dimensions are coherent
        assert_dimensions(c, d, dim)

        return new{Typ}(y, Zvar, T, R, d, c, H, Q, dim, find_missing_observations(y), "time-invariant")
    end
end

"""
    SmoothedState{T <: Real}

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `alpha` Expected value of the smoothed state ``E(\\alpha_t|y_1, \\dots , y_n)``
* `V` Error covariance matrix of smoothed state ``Var(\\alpha_t|y_1, \\dots , y_n)``
"""
struct SmoothedState{T <: Real}
    alpha::Matrix{T} # smoothed state
    V::Array{T, 3} # variance of smoothed state
end

"""
    FilterOutput{T <: Real}

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `a` Predictive state ``E(\\alpha_t|y_{t-1}, \\dots , y_1)``
* `att` Filtered state ``E(\\alpha_t|y_{t}, \\dots , y_1)``
* `v` Prediction error ``v_t = y_t − Z_t a_t``
* `P` Covariance matrix of predictive state ``P = Var(\\alpha_t|y_{t−1}, \\dots , y_1)``
* `Ptt` Covariance matrix of filtered state ``P = Var(\\alpha_t|y_{t}, \\dots , y_1)``
* `F` Variance of prediction error ``Var(v_{t})``
* `steadystate` Boolean to indicate if steady state was attained
* `tsteady` Instant when steady state was attained; in case it was not, `tsteady = n+1`
"""
struct FilterOutput{T <: Real}
    a::Matrix{T} # predictive state
    att::Matrix{T} # filtered state
    v::Matrix{T} # innovations
    P::Array{T, 3} # covariance matrix of the predictive state
    Ptt::Array{T, 3} # covariance matrix of the filtered state
    F::Array{T, 3} # covariance matrix of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
end

"""
    StateSpace{T <: Real}

A state-space structure containing the model, filter output, smoother output, covariance matrices, filter type and optimization method.
"""
struct StateSpace{T <: Real}
    model::StateSpaceModel{T}
    filter::FilterOutput{T}
    smoother::SmoothedState{T}
    filter_type::DataType
    opt_method::AbstractOptimizationMethod
end

"""
    Diagnostics{T <: Real}

A structure containing the diagnostics of a state-space model.
"""
struct Diagnostics{T <: Real}
    p_jarquebera::Vector{T}
    p_ljungbox::Vector{T}
    p_homo::Vector{T}
end
