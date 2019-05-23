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
* `Z` A vector of ``n`` ``p \\times m`` matrices of observation equations
* `T` A ``m \\times m`` matrix of the first state equations
* `R` A ``m \\times r`` matrix of the second state equations
"""
struct StateSpaceModel
    y::Matrix{Float64} # observations
    Z::Vector{Matrix{Float64}} # observation matrix
    T::Matrix{Float64} # state matrix
    R::Matrix{Float64} # state error matrix
    dim::StateSpaceDimensions
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
* `steadystate`
* `tsteady`
* `Ksteady`
* `U2star`
* `sqrtP`
* `sqrtF`
* `sqrtPsteady`
"""
mutable struct FilterOutput
    a::Vector{Matrix{Float64}} # predictive state
    v::Vector{Matrix{Float64}} # innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained
    Ksteady::Matrix{Float64}
    U2star::Vector{Matrix{Float64}}
    sqrtP::Vector{Matrix{Float64}} # lower triangular matrix with sqrt-covariance of the predictive state
    sqrtF::Vector{Matrix{Float64}} # lower triangular matrix with sqrt-covariance of the innovations
    sqrtPsteady::Matrix{Float64}
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
