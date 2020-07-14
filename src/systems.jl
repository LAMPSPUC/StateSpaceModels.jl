abstract type StateSpaceSystem end

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

get_num_states(system::LinearUnivariateTimeInvariant) = size(system.T, 1)

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

get_num_states(system::LinearUnivariateTimeVariant) = size(system.T[1], 1)
function fill_system_matrice_with_value_in_time(vector::Vector{Fl}, value::Fl) where Fl
    for t in axes(vector, 1)
        vector[t] = value
    end
end


# Functions to bridge StateSpaceSystems
"""
TODO
"""
function to_time_variant(system::LinearUnivariateTimeInvariant)

end

"""
"""
function to_multivariate(system::LinearUnivariateTimeInvariant)

end