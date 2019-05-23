"""
    StateSpaceDimensions

StateSpaceModel dimensions, following the notation of on the book 
"Time Series Analysis by State Space Methods" (2012) by J. Durbin and S. J. Koopman.

* `n` is the number of observations of the y
* `p` is the dimension of the observation vector $$y_t$$.
* `m` is the dimension of the state vector $$\alpha_t$$
* `r` is the dimension of the state covariance matrix $$Q_t$$
"""
struct StateSpaceDimensions
    n::Int
    p::Int
    m::Int
    r::Int
end

"""
    StateSpaceModel

#TODO
"""
struct StateSpaceModel
    y::VecOrMat{Float64} # observations
    Z::Vector{Matrix{Float64}} # observation matrix
    T::Matrix{Float64} # state matrix
    R::Matrix{Float64} # state error matrix
    dim::StateSpaceDimensions
end

"""
    StateSpaceParameters

#TODO
"""
mutable struct StateSpaceParameters
    sqrtH::Matrix{Float64} # lower triangular matrix with sqrt-covariance of the observation
    sqrtQ::Matrix{Float64} # lower triangular matrix with sqrt-covariance of the state
end

"""
    SmoothedState

#TODO
"""
struct SmoothedState
    alpha::Vector{Matrix{Float64}} # smoothed state
    V::Vector{Matrix{Float64}} # variance of smoothed state
end

"""
    FilterOutput

#TODO
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

#TODO
"""
struct StateSpace
    model::StateSpaceModel
    state::SmoothedState
    param::StateSpaceParameters
    filter::FilterOutput
end
