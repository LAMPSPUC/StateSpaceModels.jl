abstract type StateSpaceSystem end

"""
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

        dim_str = "Z is 1x$(mz), T is $(mt1)x$(mt2), R is $(mr)x$(rr), " *
                  "d is a number, c is $(mc)x1, H is a number. "
        
        !(mz == mt1 == mt2 == mr == mc) && throw(DimensionMismatch(dim_str))
        !(rr == rq1 == rq2) && throw(DimensionMismatch(dim_str))

        return new{Fl}(y, Z, T, R, d, c, H, Q)
    end
end

num_states(system::LinearUnivariateTimeInvariant) = size(system.T, 1)

"""
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
                                             d::Vector{Fl}, c::Vector{Vector{Fl}}, H::Vector{Fl},
                                             Q::Vector{Matrix{Fl}}) where Fl

        # TODO assert dimensions

        return new{Fl}(y, Z, T, R, d, c, H, Q)
    end
end

num_states(system::LinearUnivariateTimeVariant) = size(system.T[1], 1)

"""
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
                                                   d::Vector{Vector{Fl}}, c::Vector{Vector{Fl}}, H::Vector{Matrix{Fl}},
                                                   Q::Vector{Matrix{Fl}}) where Fl

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