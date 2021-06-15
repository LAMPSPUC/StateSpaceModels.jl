# These methods are used solely to help the auto-arima algorithm
# Kwiatkowski, D.; 
# Phillips, P. C. B.; 
# Schmidt, P.; 
# Shin, Y. (1992)
# "Testing the null hypothesis 
#  of stationarity against the 
#  alternative of a unit root"

function calc_lag_kpss(lags::Bool, n::Int)
    return lags ? trunc(Int, 4*(n/100)^0.25) : trunc(Int, 12*(n/100)^0.25)
end

function σ²_estimator(l::Int, n::Int, ϵ::Vector{Fl}) where Fl
    return ((1/n)*sum(ϵ[t]^2 for t in 1:n)) + ((2/n)*sum((1-(s/(l + 1))) * (sum(ϵ[t]*ϵ[t - s] for t in s + 1:n)) for s in 1:l))
end

function p_value_weighted_average(η::Fl, p1::Fl, cv1::Fl, 
                                  p2::Fl, cv2::Fl) where Fl
    return ((p1*(cv1 - cv2)) - ((cv1 - η)*(p1 - p2)))/(cv1 - cv2)
end

function p_value_from_η(η::Fl, crit_vals::Vector{Fl}) where Fl
    if η >= crit_vals[1]
        p_value = 0.01
    elseif η < crit_vals[1] && η >= crit_vals[2]
        p_value = p_value_weighted_average(η, 0.01, crit_vals[1], 0.025, crit_vals[2])
    elseif η < crit_vals[2] && η >= crit_vals[3]
        p_value = p_value_weighted_average(η, 0.025, crit_vals[2], 0.05, crit_vals[3])
    elseif η < crit_vals[3] && η <= crit_vals[4]
        p_value = p_value_weighted_average(η, 0.05, crit_vals[3], 0.1, crit_vals[4])
    else
        p_value = 0.1
    end
    return p_value
end

function calc_η_level(y::Vector{Fl}, lags::Bool) where Fl
    n = length(y)
    ϵ = y .- (mean(y))
    l = calc_lag_kpss(lags, n)
    η = (1/n^2)*sum(sum((((sum(ϵ[i] for i in 1:t))^2)/σ²_estimator(l, n, ϵ)) for t in 1:n))
    return η
end

function kpss_test(y::Vector{Fl}; lags::Bool = true) where Fl
    η = calc_η_level(y, lags)
    crit_vals = [0.739, 0.574, 0.463, 0.347]
    p_value = p_value_from_η(η, crit_vals)
    return p_value
end