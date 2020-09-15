export LinearUnivariateTimeInvariant,
    LinearUnivariateTimeVariant,
    LinearMultivariateTimeInvariant,
    LinearMultivariateTimeVariant

export simulate

@doc raw"""
    StateSpaceSystem

Abstract type that unifies the definition of state space models matrices such as ``y, Z, d, T, c, R, H, Q`` for
linear models.
"""
abstract type StateSpaceSystem end

@doc raw"""
    LinearUnivariateTimeInvariant

Definition of the system matrices ``y, Z, d, T, c, R, H, Q`` for linear univariate time invariant state space models.

```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  Z\alpha_{t} + d + \varepsilon_{t} \quad \varepsilon_{t} \sim \mathcal{N}(0, H)\\
        \alpha_{t+1} &= T\alpha_{t} + c + R\eta_{t} \quad \eta_{t} \sim \mathcal{N}(0, Q)\\
    \end{aligned}
\end{gather*}
```

where:

* ``y_{t}`` is a scalar
* ``Z`` is a ``m \times 1`` vector
* ``d`` is a scalar
* ``T`` is a ``m \times m`` matrix
* ``c`` is a ``m \times 1`` vector
* ``R`` is a ``m \times r`` matrix
* ``H`` is a scalar
* ``Q`` is a ``r \times r`` matrix
"""
mutable struct LinearUnivariateTimeInvariant{Fl <: Real} <: StateSpaceSystem
    y::Vector{Fl}
    Z::Vector{Fl}
    T::Matrix{Fl}
    R::Matrix{Fl}
    d::Fl
    c::Vector{Fl}
    H::Fl
    Q::Matrix{Fl}

    function LinearUnivariateTimeInvariant{Fl}(y::Vector{Fl}, Z::Vector{Fl},
                                               T::Matrix{Fl}, R::Matrix{Fl},
                                               d::Fl, c::Vector{Fl}, H::Fl,
                                               Q::Matrix{Fl}) where Fl

        mz       = length(Z)
        mt1, mt2 = size(T)
        mr, rr   = size(R)
        mc       = length(c)
        rq1, rq2 = size(Q)

        dim_str = "Z is $(mz)x1, T is $(mt1)x$(mt2), R is $(mr)x$(rr), " *
                  "d is a scalar, c is $(mc)x1, H is a scalar."

        !(mz == mt1 == mt2 == mr == mc) && throw(DimensionMismatch(dim_str))
        !(rr == rq1 == rq2) && throw(DimensionMismatch(dim_str))

        return new{Fl}(y, Z, T, R, d, c, H, Q)
    end
end

num_states(system::LinearUnivariateTimeInvariant) = size(system.T, 1)

@doc raw"""
    LinearUnivariateTimeVariant

Definition of the system matrices ``y, Z, d, T, c, R, H, Q`` for linear univariate time variant state space models.

```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  Z_{t}\alpha_{t} + d_{t} + \varepsilon_{t} \quad \varepsilon_{t} \sim \mathcal{N}(0, H_{t})\\
        \alpha_{t+1} &= T_{t}\alpha_{t} + c_{t} + R_{t}\eta_{t} \quad \eta_{t} \sim \mathcal{N}(0, Q_{t})\\
    \end{aligned}
\end{gather*}
```

where:

* ``y_{t}`` is a scalar
* ``Z_{t}`` is a ``m \times 1`` vector
* ``d_{t}`` is a scalar
* ``T_{t}`` is a ``m \times m`` matrix
* ``c_{t}`` is a ``m \times 1`` vector
* ``R_{t}`` is a ``m \times r`` matrix
* ``H_{t}`` is a scalar
* ``Q_{t}`` is a ``r \times r`` matrix
"""
mutable struct LinearUnivariateTimeVariant{Fl <: Real} <: StateSpaceSystem
    y::Vector{Fl}
    Z::Vector{Vector{Fl}}
    T::Vector{Matrix{Fl}}
    R::Vector{Matrix{Fl}}
    d::Vector{Fl}
    c::Vector{Vector{Fl}}
    H::Vector{Fl}
    Q::Vector{Matrix{Fl}}

    function LinearUnivariateTimeVariant{Fl}(y::Vector{Fl}, Z::Vector{Vector{Fl}},
                                             T::Vector{Matrix{Fl}}, R::Vector{Matrix{Fl}},
                                             d::Vector{Fl}, c::Vector{Vector{Fl}},
                                             H::Vector{Fl}, Q::Vector{Matrix{Fl}}) where Fl

        # TODO assert dimensions

        return new{Fl}(y, Z, T, R, d, c, H, Q)
    end
end

num_states(system::LinearUnivariateTimeVariant) = size(system.T[1], 1)

"""
TODO
"""
mutable struct LinearMultivariateTimeInvariant{Fl <: Real} <: StateSpaceSystem
    y::Matrix{Fl}
    Z::Matrix{Fl}
    T::Matrix{Fl}
    R::Matrix{Fl}
    d::Vector{Fl}
    c::Vector{Fl}
    H::Matrix{Fl}
    Q::Matrix{Fl}

    function LinearMultivariateTimeInvariant{Fl}(y::Matrix{Fl}, Z::Matrix{Fl},
                                                 T::Matrix{Fl}, R::Matrix{Fl},
                                                 d::Vector{Fl}, c::Vector{Fl}, H::Matrix{Fl},
                                                 Q::Matrix{Fl}) where Fl

        # TODO assert dimensions

        return new{Fl}(y, Z, T, R, d, c, H, Q)
    end
end

"""
TODO
"""
mutable struct LinearMultivariateTimeVariant{Fl <: Real} <: StateSpaceSystem
    y::Matrix{Fl}
    Z::Vector{Matrix{Fl}}
    T::Vector{Matrix{Fl}}
    R::Vector{Matrix{Fl}}
    d::Vector{Vector{Fl}}
    c::Vector{Vector{Fl}}
    H::Vector{Matrix{Fl}}
    Q::Vector{Matrix{Fl}}

    function LinearMultivariateTimeVariant{Fl}(y::Matrix{Fl}, Z::Vector{Matrix{Fl}},
                                                   T::Vector{Matrix{Fl}}, R::Vector{Matrix{Fl}},
                                                   d::Vector{Vector{Fl}}, c::Vector{Vector{Fl}},
                                                   H::Vector{Matrix{Fl}}, Q::Vector{Matrix{Fl}}) where Fl

        # TODO assert dimensions

        return new{Fl}(y, Z, T, R, d, c, H, Q)
    end
end

# Some useful functions for dealing with these systems
function fill_system_matrice_with_value_in_time(vector::Vector{Fl}, value::Fl) where Fl
    for t in axes(vector, 1)
        vector[t] = value
    end
end

function repeat_system_matrice_n_times(value::T, n::Int) where T
    vector_of_matrice = Vector{T}(undef, n)
    for i in 1:n
        vector_of_matrice[i] = value
    end
    return vector_of_matrice
end

function to_multivariate_Z(Z::Vector)
    return permutedims(Z)
end
function to_multivariate_Z(Z::Vector{Vector{Fl}}) where Fl
    multivariate_Z = Vector{Matrix{Fl}}(undef, length(Z))
    for i in 1:length(Z)
        multivariate_Z[i] = permutedims(Z[i])
    end
    return multivariate_Z
end
function to_multivariate_H(H::Fl) where Fl
    return [H][:, :]
end
function to_multivariate_H(H::Vector{Fl}) where Fl
    multivariate_H = Vector{Matrix{Fl}}(undef, length(H))
    for i in 1:length(H)
        multivariate_H[i] = permutedims([H[i]])
    end
    return multivariate_H
end
function to_multivariate_d(d::Fl) where Fl
    return [d]
end
function to_multivariate_d(d::Vector{Fl}) where Fl
    multivariate_d = Vector{Vector{Fl}}(undef, length(d))
    for i in 1:length(d)
        multivariate_d[i] = [d[i]]
    end
    return multivariate_d
end

# Bridges between Linear systems

# LinearUnivariateTimeInvariant to (LinearUnivariateTimeVariant,
#                                   LinearMultivariateTimeInvariant, TODO
#                                   LinearMultivariateTimeVariant)
function to_univariate_time_variant(system::LinearUnivariateTimeInvariant{Fl}) where Fl
    n = length(system.y)
    y = system.y
    Z = repeat_system_matrice_n_times(system.Z, n)
    T = repeat_system_matrice_n_times(system.T, n)
    R = repeat_system_matrice_n_times(system.R, n)
    d = repeat_system_matrice_n_times(system.d, n)
    c = repeat_system_matrice_n_times(system.c, n)
    H = repeat_system_matrice_n_times(system.H, n)
    Q = repeat_system_matrice_n_times(system.Q, n)
    return LinearUnivariateTimeVariant{Fl}(y, Z, T, R, d, c, H, Q)
end
function to_multivariate_time_variant(system::LinearUnivariateTimeInvariant)
    univariate_time_variant = to_univariate_time_variant(system)
    return to_multivariate_time_variant(univariate_time_variant)
end

# LinearUnivariateTimeVariant to (LinearMultivariateTimeVariant)
function to_multivariate_time_variant(system::LinearUnivariateTimeVariant{Fl}) where Fl
    y = system.y[:, :]
    Z = to_multivariate_Z(system.Z)
    T = system.T
    R = system.R
    d = to_multivariate_d(system.d)
    c = system.c
    H = to_multivariate_H(system.H)
    Q = system.Q
    return LinearMultivariateTimeVariant{Fl}(y, Z, T, R, d, c, H, Q)
end

to_multivariate_time_variant(system::LinearMultivariateTimeVariant) = system

# Functions for simulations
function simulate(sys::LinearUnivariateTimeInvariant{Fl},
                  initial_state::Vector{Fl},
                  n::Int;
                  return_simulated_states::Bool=false) where Fl

    m = size(sys.T, 1)
    y = Vector{Fl}(undef, n)
    alpha = Matrix{Fl}(undef, n + 1, m)
    # Sampling errors
    chol_H     = sqrt(sys.H)
    chol_Q     = cholesky(sys.Q)
    standard_ε = randn(n)
    standard_η = randn(n + 1, size(sys.Q, 1))

    # The first state of the simulation is the update of a_0
    alpha[1, :] = sys.T * initial_state + sys.c + sys.R * chol_Q.L * standard_η[1, :]
    # Simulate scenarios
    for t = 1:n
        y[t]            = dot(sys.Z, alpha[t, :]) + sys.d + chol_H * standard_ε[t]
        alpha[t + 1, :] = sys.T * alpha[t, :] + sys.c + sys.R * chol_Q.L * standard_η[t, :]
    end

    if return_simulated_states
        return y, alpha[1:n, :]
    end
    return y
end
