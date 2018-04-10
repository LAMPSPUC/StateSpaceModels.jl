"""Structure with state space dimensions"""
struct StateSpaceDimensions
    n::Int
    p::Int
    m::Int
    r::Int
    p_exp::Int
end

"""Structure with state space matrices and data"""
struct StateSpaceSystem
    y::Array # observations
    X::Array # exogenous variables
    s::Int # seasonality
    Z::Array # observation matrix
    T::Array # state matrix
    R::Array # state error matrix
end

"""Structure with state space hyperparameters"""
mutable struct StateSpaceParameters
    sqrtH::Array # lower triangular matrix with sqrt-covariance of the observation
    sqrtQ::Array # lower triangular matrix with sqrt-covariance of the state
end

"""Structure with smoothed state"""
struct SmoothedState
    trend::Array # smoothed trend
    slope::Array # smoothed slope
    seasonal::Array # smoothed seasonality
    exogenous::Array # smoothed regression of exogenous variables
    V::Array # variance of smoothed state
    alpha::Array # smoothed state matrix
end

"""Structure with Kalman filter output"""
mutable struct FilterOutput
    a::Array # predictive state
    v::Array # innovations
    steadystate::Bool # flag that indicates if steady state was attained
    tsteady::Int # instant when steady state was attained
    Ksteady::Array
    U2star::Array
    sqrtP::Array # lower triangular matrix with sqrt-covariance of the predictive state
    sqrtF::Array # lower triangular matrix with sqrt-covariance of the innovations
    sqrtPsteady::Array
end

"""General output structure for the user"""
struct StateSpace
    sys::StateSpaceSystem
    dim::StateSpaceDimensions
    state::SmoothedState
    param::StateSpaceParameters
    filter::FilterOutput
end
