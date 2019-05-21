abstract type StateSpaceModel end

"""
    BasicStructuralModel <: StateSpaceModel

Basic structural model: stochastic trend, slope and seasonality
"""
struct BasicStructuralModel <: StateSpaceModel
    y::Matrix{T} where T <: AbstractFloat
    s::Int
end

"""
    StructuralModelExogenous <: StateSpaceModel

Structural model with exogenous variables and stochastic trend, slope and seasonality
"""
struct StructuralModelExogenous <: StateSpaceModel
    y::Matrix{T} where T <: AbstractFloat
    s::Int
    X::Matrix{T} where T <: AbstractFloat
end

"""
    StructuralModel(y::VecOrMat{T}, s::Int; X::VecOrMat{T} = Matrix{T}(undef, 0, 0)) where T <: AbstractFloat

Pseudo-constructor for structural models
"""
function StructuralModel(y::VecOrMat{T}, s::Int; X::VecOrMat{T} = Matrix{Float64}(undef, 0, 0)) where T <: AbstractFloat
    if isempty(X)
        BasicStructuralModel(y[:, :], s)
    else
        StructuralModelExogenous(y[:, :], s, X[:, :])
    end
end