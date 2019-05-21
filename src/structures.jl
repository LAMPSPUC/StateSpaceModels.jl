"""Structure with state space dimensions"""
struct StateSpaceDimensions
    n::Int
    p::Int
    m::Int
    r::Int
end

"""Structure with state space matrices and data"""
struct StateSpaceSystem
    y::Matrix{Float64} # observations
    Z::Vector{Matrix{Float64}} # observation matrix
    T::Matrix{Float64} # state matrix
    R::Matrix{Float64} # state error matrix
    dim::StateSpaceDimensions
end

"""Structure with state space hyperparameters"""
mutable struct StateSpaceParameters
    sqrtH::Matrix{Float64} # lower triangular matrix with sqrt-covariance of the observation
    sqrtQ::Matrix{Float64} # lower triangular matrix with sqrt-covariance of the state
end

"""Structure with smoothed state"""
struct SmoothedState
    alpha::Vector{Matrix{Float64}} # smoothed state
    V::Vector{Matrix{Float64}} # variance of smoothed state
end

"""Structure with Kalman filter output"""
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

"""General output structure for the user"""
struct StateSpace
    sys::StateSpaceSystem
    state::SmoothedState
    param::StateSpaceParameters
    filter::FilterOutput
end
