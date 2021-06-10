# Seasonality and unobserved components
# models: an overview
# Andrew Harvey
# December 4, 2006

# Are Seasonal Patterns Constant Over Time?
# A Test for Seasonal Stability 
# Fabio CANO
# Bruce E. HANSE
# Journal of Business & Economic Statistics, July 1995, Vol. 13, No. 3

λ(j::Int, s::Int) = (2*pi*j)/s

const CRIT_VALS_05 = [0.470, 0.749, 1.01, 1.24, 1.47, 1.68, 1.9, 2.11, 2.32, 2.54, 2.75, 2.96]

function crit_val_05_generalized_von_mises_distribution(s::Int)
    if s > 12
        return 0.269*s*0.928
    else
        return CRIT_VALS_05[s]
    end
end

function π_statistic(ε::Vector{Fl}, σ::Fl, T::Int) where Fl
    return (T^(-2)*σ^(-2))*
            sum(
                sum(ε[i]*(-1)^(i) for i = 1:t) for t = 1:T
            )
end

function std_statistic(ε::Vector{Fl}, σ::Fl, T::Int, s::Int, j::Int) where Fl
    return (2*T^(-2)*σ^(-2))*
            sum(
                sum(ε[i]*cos(λ(j, s)*i) for i = 1:t)^2 
                +
                sum(ε[i]*sin(λ(j, s)*i) for i = 1:t)^2 for t = 1:T
            )
end

function test_statistic(ε::Vector{Fl}, σ::Fl, T::Int, s::Int) where Fl
    ω = Float64[]
    for j = 1:Int(floor(s/2)) #check
        ω_j = j == s/2 ? π_statistic(ε, σ, T) : std_statistic(ε, σ, T, s, j)
        push!(ω, ω_j)
    end
    return sum(ω)
end

function build_std(T::Int, s::Int, j::Int)
    t = collect(1:T)
    return [cos.(λ(j, s).*t) sin.(λ(j, s).*t)]
end

function build_π(T::Int, s::Int, j::Int)
    t = collect(1:T)
    return [cos.(λ(j, s)*t) ones(T)]
end

function build_X(s::Int, T::Int)
    X = Array{Float64, 2}(undef, T, s)
    for j = 1:Int(floor(s/2))
        X[:,[2*j-1, 2*j]] = j == s/2 ? build_π(T, s, j) : build_std(T, s, j)
    end
    return X
end

function seasonal_regression(y::Vector{Fl}, s::Int, T::Int) where Fl
    X = build_X(s, T)
    return y - X \ y * y
end

function seasonal_stationarity_test(y::Vector{Fl}, s::Int) where Fl
    T  = length(y)
    ε  = seasonal_regression(y, s, T)
    σ  = std(ε)
    ω  = test_statistic(ε, σ, T, s)
    crit_val = crit_val_05_generalized_von_mises_distribution(s)
    res = ω - crit_val
    res > 0 ? println("Rejected seasonal stationarity at 5% significance level"*
            " Test Statistic: $ω - Critical Value: $crit_val") :
              println("Didn't rejected seasonal stationarity at 5% significance level"*
            " Test Statistic: $ω - Critical Value: $crit_val")
    return res > 0 ? false : true
end

