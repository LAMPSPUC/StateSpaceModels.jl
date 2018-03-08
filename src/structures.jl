"""State space system dimensions"""
struct StateSpaceDimensions
    n::Int
    p::Int
    m::Int
    r::Int
    p_exp::Int
end

"""State space data matrices"""
struct StateSpaceSystem
    y::Array # observations
    X::Array # exogenous variables
    s::Int # seasonality period
    Z::Array # observation matrix
    T::Array # state matrix
    R::Array # state noise matrix
end

"""Fixed parameters of the state space system"""
mutable struct StateSpaceParameters
    sqrtH::Array
    sqrtQ::Array
end

"""Smoothed state data"""
struct SmoothedState
    trend::Array # trend
    slope::Array # slope
    seasonal::Array # seasonal
    V::Array # state variance
    alpha::Array # state matrix
end

"""Kalman Filter output"""
mutable struct FilterOutput
    a::Array
    v::Array
    steadystate::Bool
    tsteady::Int
    Ksteady::Array
    U2star::Array
    sqrtP::Array
    sqrtF::Array
    sqrtPsteady::Array
end

"""Output data structure"""
struct StateSpace
    sys::StateSpaceSystem # system matrices
    dim::StateSpaceDimensions # system dimensions
    state::SmoothedState # smoothed state
    param::StateSpaceParameters # fixed parameters
    steadystate::Bool # flag that indicates if steady state was attained
end
