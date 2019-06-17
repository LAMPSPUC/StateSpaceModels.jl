export StateSpaceDimensions, StateSpaceModel, StateSpaceCovariance, 
       SmoothedState, FilteredState, StateSpace

# Abstract types
"""
TODO
"""
abstract type AbstractFilter end

"""
TODO
"""
abstract type AbstractSmoother end

"""
TODO
"""
abstract type AbstractOptimizationMethod end

# Auxiliary structures to square root kalman filter and smoother
mutable struct SquareRootFilter <: AbstractFilter
    a::Matrix{Float64} # predictive state
    v::Matrix{Float64} # innovations
    sqrtP::Array{Float64, 3} # lower triangular matrix with sqrt-covariance of the predictive state
    sqrtF::Array{Float64, 3} # lower triangular matrix with sqrt-covariance of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
    K::Array{Float64, 3} # Kalman gain
    U2star::Array{Float64, 3} # Auxiliary sqrtKalman filter matrix
end

mutable struct SquareRootSmoother <: AbstractSmoother
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

    function RandomSeedsLBFGS()
        return new(1e-10, 1e-10, 1e5, 3)
    end
end




"""
    StateSpaceDimensions

StateSpaceModel dimensions, following the notation of on the book 
\"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `n` is the number of observations
* `p` is the dimension of the observation vector ``y_t``
* `m` is the dimension of the state vector ``\\alpha_t``
* `r` is the dimension of the state covariance matrix ``Q_t``
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
    mode::String
    filter_type::DataType
    optimization_method::AbstractOptimizationMethod

    function StateSpaceModel(y::Matrix{Float64}, Z::Array{Float64, 3}, T::Matrix{Float64}, R::Matrix{Float64}; 
                             filter_type::DataType = SquareRootFilter, 
                             optimization_method::AbstractOptimizationMethod = RandomSeedsLBFGS())
        
        # Validate StateSpaceDimensions
        ny, py = size(y)
        pz, mz, nz = size(Z)
        mt1, mt2 = size(T)
        mr, rr = size(R)
        if !((mz == mt1 == mt2 == mr) && (pz == py))
            error("StateSpaceModel dimension mismatch")
        end
        dim = StateSpaceDimensions(ny, py, mr, rr)
        new(y, Z, T, R, dim, "time-variant", filter_type, optimization_method)
    end
    
    function StateSpaceModel(y::Matrix{Float64}, Z::Matrix{Float64}, T::Matrix{Float64}, R::Matrix{Float64};
                             filter_type::DataType = SquareRootFilter, 
                             optimization_method::AbstractOptimizationMethod = RandomSeedsLBFGS())

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
        for t = 1:ny
            Zvar[:, :, t] = Z
        end
        new(y, Zvar, T, R, dim, "time-invariant", filter_type, optimization_method)
    end
end

"""
    StateSpaceCovariance

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `H` covariance matrix of the observation vector ``H_t``
* `Q` covariance matrix of the state vector ``Q_t``
"""
struct StateSpaceCovariance
    H::Matrix{Float64} 
    Q::Matrix{Float64}

    function StateSpaceCovariance(sqrtH::Matrix{Float64}, sqrtQ::Matrix{Float64}, 
                                  filter_type::Type{SquareRootFilter})
        return new(gram(sqrtH), gram(sqrtQ))
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
    FilteredState

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `a`
* `v` 
* `P`
* `F`
* `steadystate` Boolean to indicate if steady state was attained
* `tstady` Instant when steady state was attained; in case it wasn't, `tsteady = n+1`
"""
struct FilteredState
    a::Matrix{Float64} # predictive state
    v::Matrix{Float64} # innovations
    P::Array{Float64, 3} # lower triangular matrix with sqrt-covariance of the predictive state
    F::Array{Float64, 3} # lower triangular matrix with sqrt-covariance of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
end

"""
    StateSpace

StateSpaceModel
"""
struct StateSpace
    model::StateSpaceModel
    filter::FilteredState
    smoother::SmoothedState
    covariance::StateSpaceCovariance
end
