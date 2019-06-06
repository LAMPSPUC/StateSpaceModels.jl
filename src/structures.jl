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
        new(y, Z, T, R, dim, "time-variant")
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
        for t = 1:ny
            Zvar[:, :, t] = Z
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
    alpha::Matrix{Float64} # smoothed state
    V::Array{Float64, 3} # variance of smoothed state
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
    a::Matrix{Float64} # predictive state
    v::Matrix{Float64} # innovations
    sqrtP::Array{Float64, 3} # lower triangular matrix with sqrt-covariance of the predictive state
    sqrtF::Array{Float64, 3} # lower triangular matrix with sqrt-covariance of the innovations
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


mutable struct AuxiliarySqrtKalman

    a::Matrix{Float64}
    sqrtP::Array{Float64, 3}
    v::Matrix{Float64}
    sqrtF::Array{Float64, 3}
    K::Array{Float64, 3}
    U::Array{Float64, 2}
    G::Array{Float64, 2}
    U2star::Array{Float64, 3}
    zeros_pr::Array{Float64, 2}
    zeros_mp::Array{Float64, 2}
    range1::UnitRange{Int64}
    range2::UnitRange{Int64}
    sqrtH_zeros_pr::Array{Float64, 2}
    zeros_mp_RsqrtQ::Array{Float64, 2}

    function AuxiliarySqrtKalman(model::StateSpaceModel)
        aux = new()
        # Load dimensions
        n, p, m, r = size(model)
        # Predictive state and its sqrt-covariance
        aux.a     = Matrix{Float64}(undef, n+1, m)
        aux.sqrtP = Array{Float64, 3}(undef, m, m, n+1)
        # Innovation and its sqrt-covariance
        aux.v     = Matrix{Float64}(undef, n, p)
        aux.sqrtF = Array{Float64, 3}(undef, p, p, n)
        # Kalman gain
        aux.K     = Array{Float64, 3}(undef, m, p, n)
        # Auxiliary matrices
        aux.U = Array{Float64, 2}(undef, p + m, m + p + r)
        aux.G = Array{Float64, 2}(undef, m + p + r, p + m)
        aux.U2star = Array{Float64, 3}(undef, m, p, n)
        # Pre-allocating for performance
        aux.zeros_pr = zeros(Float64, p, r)
        aux.zeros_mp = zeros(Float64, m, p)
        aux.range1   = (p + 1):(p + m)
        aux.range2   = 1:p
        aux.sqrtH_zeros_pr  = zeros(Float64, p, p + r)
        aux.zeros_mp_RsqrtQ = zeros(Float64, m, p + m)

        return aux
    end
end
