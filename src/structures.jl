export StateSpaceDimensions, StateSpaceModel, StateSpaceParameters, 
       SmoothedState, FilterOutput, StateSpace

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
* `Z` A vector of dimension ``n`` where each entry is a ``p \\times m`` matrix
* `T` A ``m \\times m`` matrix
* `R` A ``m \\times r`` matrix

A `StateSpaceModel` object can be defined using `StateSpaceModel(y::Matrix{Float64}, Z::Vector{Matrix{Float64}}, T::Matrix{Float64}, R::Matrix{Float64}, dim::StateSpaceDimensions, mode::String)`.

Alternatively, if `Z` is time-invariant, it can be input as a single ``p \\times m`` matrix.
"""
struct StateSpaceModel
    y::Matrix{Float64} # observations
    Z::Vector{Matrix{Float64}} # observation matrix
    T::Matrix{Float64} # state matrix
    R::Matrix{Float64} # state error matrix
    dim::StateSpaceDimensions
    mode::String

    function StateSpaceModel(y::Matrix{Float64}, Z::Vector{Matrix{Float64}}, T::Matrix{Float64}, R::Matrix{Float64}, 
                        dim::StateSpaceDimensions, mode::String)
        
        # Validate StateSpaceDimensions
        ny, py = size(y)
        pz, mz = size(Z[1])
        mt, mt = size(T)
        mr, rr = size(R)
        if !((mm == mt == mr) && (pz = py))
            error("StateSpaceModel dimension mismatch")
        end
        dim = StateSpaceDimensions(ny, py, mr, rr)
        new(y, Z, T, R, dim, "time-variant")
    end
    
    function StateSpaceModel(y::Matrix{Float64}, Z::Matrix{Float64}, T::Matrix{Float64}, R::Matrix{Float64}, 
                        dim::StateSpaceDimensions, mode::String)

        # Validate StateSpaceDimensions
        ny, py = size(y)
        pz, mz = size(Z)
        mt, mt = size(T)
        mr, rr = size(R)
        if !((mm == mt == mr) && (pz = py))
            error("StateSpaceModel dimension mismatch")
        end
        dim = StateSpaceDimensions(ny, py, mr , rr)

        # Build Z
        Zvar = Vector{Matrix{Float64}}(undef, ny)
        for t = 1:ny
            Zvar[t] = Z
        end
        new(y, Zvar, T, R, dim, "time-invariant")
    end
end

"""
    StateSpaceParameters

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `sqrtH` matrix with sqrt-covariance of the observation vector ``H_t``
* `sqrtQ` matrix with sqrt-covariance of the state vector ``Q_t``
"""
mutable struct StateSpaceParameters
    sqrtH::Matrix{Float64} # lower triangular matrix with sqrt-covariance of the observation
    sqrtQ::Matrix{Float64} # lower triangular matrix with sqrt-covariance of the state
end

"""
    SmoothedState

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `alpha` Expected value of the smoothed state ``E(\\alpha_t|y_1, \\dots , y_n)``
* `V` Error covariance matrix of smoothed state ``Var(\\alpha_t|y_1, \\dots , y_n)``
"""
struct SmoothedState
    alpha::Vector{Matrix{Float64}} # smoothed state
    V::Vector{Matrix{Float64}} # variance of smoothed state
end

"""
    FilterOutput

Following the notation of on the book \"Time Series Analysis by State Space Methods\" (2012) by J. Durbin and S. J. Koopman.

* `a` 
* `v` 
* `sqrtP`
* `sqrtF`
* `steadystate`
"""
mutable struct FilterOutput
    a::Vector{Matrix{Float64}} # predictive state
    v::Vector{Matrix{Float64}} # innovations
    sqrtP::Vector{Matrix{Float64}} # lower triangular matrix with sqrt-covariance of the predictive state
    sqrtF::Vector{Matrix{Float64}} # lower triangular matrix with sqrt-covariance of the innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained; in case it wasn't, tsteady = n+1
end

"""
    StateSpace

StateSpaceModel
"""
struct StateSpace
    model::StateSpaceModel
    state::SmoothedState
    param::StateSpaceParameters
    filter::FilterOutput
end
