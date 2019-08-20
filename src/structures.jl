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
mutable struct SquareRootFilter <: AbstractFilter
    a::Matrix{Float64} # predictive state
    v::Matrix{Float64} # innovations
    sqrtP::Array{Float64, 3} # lower triangular matrix with sqrt-covariance of the predictive state
    sqrtF::Array{Float64, 3} # lower triangular matrix with sqrt-covariance of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
    K::Array{Float64, 3} # Kalman gain
end

# Auxiliary structure for Kalman filter
mutable struct KalmanFilter <: AbstractFilter
    a::Matrix{Float64} # predictive state
    att::Matrix{Float64}
    v::Matrix{Float64} # innovations
    P::Array{Float64, 3} # covariance matrix of the predictive state
    Ptt::Array{Float64, 3}
    F::Array{Float64, 3} # covariance matrix of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
    K::Array{Float64, 3} # Kalman gain
end

# Auxiliary structure for univariate Kalman filter
mutable struct UnivariateKalmanFilter <: AbstractFilter
    a::Matrix{Float64} # predictive state
    att::Matrix{Float64}
    v::Vector{Float64} # innovations
    P::Array{Float64, 3} # covariance matrix of the predictive state
    Ptt::Array{Float64, 3}
    F::Vector{Float64} # covariance matrix of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
    K::Matrix{Float64} # Kalman gain
end

# Auxiliary structure for smoother
mutable struct Smoother <: AbstractSmoother
    alpha::Matrix{Float64} # smoothed state
    V::Array{Float64, 3} # variance of smoothed state
end


# Concrete structs for optimization methods
"""
TODO
"""
mutable struct RandomSeedsLBFGS <: AbstractOptimizationMethod
    f_tol::Float64
    g_tol::Float64
    iterations::Int
    nseeds::Int
    seeds::Array{Float64}

    function RandomSeedsLBFGS(; f_tol::Float64 = 1e-6, g_tol::Float64 = 1e-6, iterations::Int = 10^5, nseeds::Int = 3)
        return new(f_tol, g_tol, 1e5, nseeds)
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
    StateSpaceModel

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `y` A ``n \\times p`` matrix containing observations
* `Z` A ``p \\times m \\times n`` matrix
* `T` A ``m \\times m`` matrix
* `R` A ``m \\times r`` matrix

A `StateSpaceModel` object can be defined using `StateSpaceModel(y::Matrix{Float64}, Z::Array{Float64, 3}, T::Matrix{Float64}, R::Matrix{Float64})`.

Alternatively, if `Z` is time-invariant, it can be input as a single ``p \\times m`` matrix.
"""
struct StateSpaceModel
    y::Matrix{Float64} # observations
    Z::Array{Float64, 3} # observation matrix
    T::Matrix{Float64} # state matrix
    R::Matrix{Float64} # state error matrix
    dim::StateSpaceDimensions
    missing_observations::Vector{Int}
    mode::String

    function StateSpaceModel(y::Matrix{Float64}, Z::Array{Float64, 3}, T::Matrix{Float64}, R::Matrix{Float64})

        # Validate StateSpaceDimensions
        ny, py = size(y)
        pz, mz, nz = size(Z)
        mt1, mt2 = size(T)
        mr, rr = size(R)
        if !((mz == mt1 == mt2 == mr) && (pz == py))
            error("StateSpaceModel dimension mismatch")
        end
        dim = StateSpaceDimensions(ny, py, mr, rr)
        new(y, Z, T, R, dim, find_missing_observations(y), "time-variant")
    end

    function StateSpaceModel(y::Matrix{Float64}, Z::Matrix{Float64}, T::Matrix{Float64}, R::Matrix{Float64})

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
        Zvar = Array{Float64, 3}(undef, pz, mz, ny)
        for t in 1:ny, i in axes(Z, 1), j in axes(Z, 2)
            Zvar[i, j, t] = Z[i, j] # Zvar[:, :, t] = Z
        end
        new(y, Zvar, T, R, dim, find_missing_observations(y), "time-invariant")
    end
end

"""
    StateSpaceCovariance

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `H` Covariance matrix of the observation vector
* `Q` Covariance matrix of the state vector
"""
struct StateSpaceCovariance
    H::Matrix{Float64}
    Q::Matrix{Float64}

    function StateSpaceCovariance(sqrtH::Matrix{Float64}, sqrtQ::Matrix{Float64},
                                  filter_type::Type{SquareRootFilter})
        return new(gram(sqrtH), gram(sqrtQ))
    end

    function StateSpaceCovariance(H::Matrix{Float64}, Q::Matrix{Float64},
                                    filter_type::Type{KalmanFilter})
        return new(H, Q)
    end
end

"""
    SmoothedState

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `alpha` Expected value of the smoothed state ``E(\\alpha_t|y_1, \\dots , y_n)``
* `V` Error covariance matrix of smoothed state ``Var(\\alpha_t|y_1, \\dots , y_n)``
"""
struct SmoothedState
    alpha::Matrix{Float64} # smoothed state
    V::Array{Float64, 3} # variance of smoothed state
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
struct FilterOutput
    a::Matrix{Float64} # predictive state
    att::Matrix{Float64} # filtered state
    v::Matrix{Float64} # innovations
    P::Array{Float64, 3} # covariance matrix of the predictive state
    Ptt::Array{Float64, 3} # covariance matrix of the filtered state
    F::Array{Float64, 3} # covariance matrix of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
end

"""
    StateSpace

A state-space structure containing the model, filter output, smoother output, covariance matrices, filter type and optimization method.
"""
struct StateSpace
    model::StateSpaceModel
    filter::FilterOutput
    smoother::SmoothedState
    covariance::StateSpaceCovariance
    filter_type::DataType
    optimization_method::AbstractOptimizationMethod
end
