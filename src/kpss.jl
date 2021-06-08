function calc_lag_kpss(lags::Bool, n::Int64)
    return lags ? trunc(Int, 4*(n/100)^0.25) : trunc(Int, 12*(n/100)^0.25)
end

function σ²_estimator(l::Int64, n::Int64, ϵ::Vector{Fl}) where Fl
    return ((1/n)*sum(ϵ[t]^2 for t in 1:n)) + ((2/n)*sum((1-(s/(l + 1))) * (sum(ϵ[t]*ϵ[t - s] for t in s + 1:n)) for s in 1:l))
end

function p_value_weighted_average(η::Float64, p1::Float64, cv1::Float64, 
                                  p2::Float64, cv2::Float64)
    return ((p1*(cv1 - cv2)) - ((cv1 - η)*(p1 - p2)))/(cv1 - cv2)
end

function p_value_from_η(η::Float64, crit_vals::Vector{Float64})
    if η >= crit_vals[1]
        p_value = 0.01
        @warn("p-value smaller than printed p-value")
    elseif η < crit_vals[1] && η >= crit_vals[2]
        p_value = p_value_weighted_average(η, 0.01, crit_vals[1], 0.025, crit_vals[2])
    elseif η < crit_vals[2] && η >= crit_vals[3]
        p_value = p_value_weighted_average(η, 0.025, crit_vals[2], 0.05, crit_vals[3])
    elseif η < crit_vals[3] && η <= crit_vals[4]
        p_value = p_value_weighted_average(η, 0.05, crit_vals[3], 0.1, crit_vals[4])
    else
        p_value = 0.1
        @warn("p-value greater than printed p-value")
    end
    return p_value
end

function kpss_test_trend(y::Vector{Fl}, lags::Bool) where Fl
    η = calc_η_trend(y, lags)
    crit_vals = [0.216, 0.176, 0.146, 0.119]
    p_value = p_value_from_η(η, crit_vals)

    return p_value
end

function kpss_test_level(y::Vector{Fl}, lags::Bool) where Fl
    η = calc_η_level(y, lags)
    crit_vals = [0.739, 0.574, 0.463, 0.347]
    p_value = p_value_from_η(η, crit_vals)

    return p_value
end

function calc_η_trend(y::Vector{Fl}, lags::Bool) where Fl
    n     = length(y)
    zt    = collect(1:n)
    r     = ones(n)
    x     = hcat(r,zt)
    β_est = (x'*x)\x'*y
    ŷ     = x*β_est
    ϵ     = y - ŷ
    l     = calc_lag_kpss(lags, n)
    η     = (1/n^2)*sum(sum((((sum(ϵ[i] for i in 1:t))^2)/σ²_estimator(l, n, ϵ)) for t in 1:n))

    return η
end

function calc_η_level(y::Vector{Fl}, lags::Bool) where Fl
    n     = length(y)
    ϵ     = y .- (mean(y))
    l     = calc_lag_kpss(lags, n)
    η     = (1/n^2)*sum(sum((((sum(ϵ[i] for i in 1:t))^2)/σ²_estimator(l, n, ϵ)) for t in 1:n))
    
    return η
end


function kpss_test(y::Vector{Fl}; null::Symbol = :level, lags::Bool = true) where Fl
    # Denis Kwiatkowski
    # Peter C.B. Phillips
    # Peter Schmidt and Yongcheol Shin
    # "Testing the null hypothesis of stationarity
    #  against the alternative of a unit root" 
        #    Biometrika 71 (2) (August 1): 403-404.
    # Journal of Econometrics 54 (1992) 159-178. North-Holland

    @assert null in [:trend, :level]
    
    return null == :trend ? kpss_test_trend(y, lags) : kpss_test_level(y, lags)
end