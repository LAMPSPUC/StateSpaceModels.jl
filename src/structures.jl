export StateSpaceDimensions, StateSpaceModel, StateSpaceCovariance, SmoothedState,
        FilterOutput, StateSpace

export KalmanFilter, SquareRootFilter, UnivariateKalmanFilter
export RandomSeedsLBFGS
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
    AbstractOptimizationMethod

Abstract type used to implement an interface for generic optimization methods.
"""
abstract type AbstractOptimizationMethod end

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

# Concrete structs for optimization methods
"""
TODO
"""
mutable struct RandomSeedsLBFGS{T <: Real} <: AbstractOptimizationMethod
    f_tol::T
    g_tol::T
    iterations::Int
    nseeds::Int
    seeds::Array{T}

    function RandomSeedsLBFGS(; f_tol::T = 1e-6, g_tol::T = 1e-6, iterations::Int = 10^5, nseeds::Int = 3) where T <: Real
        return new{T}(f_tol, g_tol, 1e5, nseeds)
    end
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
    StateSpaceModel{Typ}

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `y` A ``n \\times p`` matrix containing observations
* `Z` A ``p \\times m \\times n`` matrix
* `T` A ``m \\times m`` matrix
* `R` A ``m \\times r`` matrix

A `StateSpaceModel` object can be defined using `StateSpaceModel(y::Matrix{Typ}, Z::Array{Typ, 3}, T::Matrix{Typ}, R::Matrix{Typ})`.

Alternatively, if `Z` is time-invariant, it can be input as a single ``p \\times m`` matrix.
"""
struct StateSpaceModel{Typ <: Real}
    y::Matrix{Typ} # observations
    Z::Array{Typ, 3} # observation matrix
    T::Matrix{Typ} # state matrix
    R::Matrix{Typ} # state error matrix
    dim::StateSpaceDimensions
    missing_observations::Vector{Int}
    mode::String

    function StateSpaceModel(y::Matrix{Typ}, Z::Array{Typ, 3}, T::Matrix{Typ}, R::Matrix{Typ}) where Typ <: Real

        # Validate StateSpaceDimensions
        ny, py = size(y)
        pz, mz, nz = size(Z)
        mt1, mt2 = size(T)
        mr, rr = size(R)
        if !((mz == mt1 == mt2 == mr) && (pz == py))
            error("StateSpaceModel dimension mismatch")
        end
        dim = StateSpaceDimensions(ny, py, mr, rr)
        new{Typ}(y, Z, T, R, dim, find_missing_observations(y), "time-variant")
    end

    function StateSpaceModel(y::Matrix{Typ}, Z::Matrix{Typ}, T::Matrix{Typ}, R::Matrix{Typ}) where Typ <: Real

        # Validate StateSpaceDimensions
        ny, py = size(y)
        pz, mz = size(Z)
        mt, mt = size(T)
        mr, rr = size(R)
        if !((mz == mt == mr) && (pz == py))
            error("StateSpaceModel dimension mismatch")
        end
        dim = StateSpaceDimensions(ny, py, mr, rr)

        # Build Z
        Zvar = Array{Typ, 3}(undef, pz, mz, ny)
        for t in 1:ny, i in axes(Z, 1), j in axes(Z, 2)
            Zvar[i, j, t] = Z[i, j] # Zvar[:, :, t] = Z
        end
        new{Typ}(y, Zvar, T, R, dim, find_missing_observations(y), "time-invariant")
    end
end

"""
    StateSpaceCovariance

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `H` Covariance matrix of the observation vector
* `Q` Covariance matrix of the state vector
"""
struct StateSpaceCovariance{T <: Real}
    H::Matrix{T}
    Q::Matrix{T}

    function StateSpaceCovariance(sqrtH::Matrix{T}, sqrtQ::Matrix{T},
                                  filter_type::Type{SquareRootFilter{T}}) where T <: Real
        return new{T}(gram(sqrtH), gram(sqrtQ))
    end

    function StateSpaceCovariance(H::Matrix{T}, Q::Matrix{T},
                                  filter_type::Union{Type{KalmanFilter{T}}, 
                                                     Type{UnivariateKalmanFilter{T}}}) where T <: Real
        return new{T}(H, Q)
    end
end

"""
    SmoothedState

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `alpha` Expected value of the smoothed state ``E(\\alpha_t|y_1, \\dots , y_n)``
* `V` Error covariance matrix of smoothed state ``Var(\\alpha_t|y_1, \\dots , y_n)``
"""
struct SmoothedState{T <: Real}
    alpha::Matrix{T} # smoothed state
    V::Array{T, 3} # variance of smoothed state
end

"""
    FilterOutput

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
    StateSpace

A state-space structure containing the model, filter output, smoother output, covariance matrices, filter type and optimization method.
"""
struct StateSpace{T <: Real}
    model::StateSpaceModel{T}
    filter::FilterOutput{T}
    smoother::SmoothedState{T}
    covariance::StateSpaceCovariance{T}
    filter_type::DataType
    optimization_method::AbstractOptimizationMethod
end

"""
    Diagnostics

A structure containing the diagnostics of a state-space model.
"""
struct Diagnostics{T <: Real}
    p_jarquebera::Vector{T}
    p_ljungbox::Vector{T}
    p_homo::Vector{T}
end