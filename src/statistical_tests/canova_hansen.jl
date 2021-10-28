# Seasonality and unobserved components
# models: an overview
# Andrew Harvey
# December 4, 2006

# Are Seasonal Patterns Constant Over Time?
# A Test for Seasonal Stability 
# Fabio CANOVA
# Bruce E. HANSEN
# Journal of Business & Economic Statistics, July 1995, Vol. 13, No. 3

λ(j::Int, s::Int) = (2*pi*j)/s

const CRIT_VALS_05 = [0.470, 0.749, 1.01, 1.24, 1.47, 1.68, 1.9, 2.11, 2.32, 2.54, 2.75, 2.96]

function crit_val_05_generalized_von_mises_distribution(s::Int)
    return s > 12 ? 0.269 * s^0.928 : CRIT_VALS_05[s]
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
    # joint statistical test for all frequencies in 
    # trigonometric pattern
    ω = 0
    # for each frequency in the trigonometric pattern
    for j = 1:floor(Int, s/2)
        ω += j == s/2 ? π_statistic(ε, σ, T) : std_statistic(ε, σ, T, s, j)
    end
    return ω
end

function build_std(T::Int, s::Int, j::Int)
    # add cossine and sine with frequency (2*pi*j)/s
    # to exogenous matrix X
    return hcat(cos.(λ(j, s)*(1:T)), sin.(λ(j, s)*(1:T)))
end

function build_π(T::Int, s::Int, j::Int)
    # when j = s/2
    # add cossine with frequency pi and intercept
    # to exogenous matrix X
    return hcat(cos.(λ(j, s)*(1:T)), ones(T))
end

function build_X(s::Int, T::Int, Fl::Type)
    # create the exogenous matrix 
    X = Array{Fl, 2}(undef, T, s)
    # for each frequency in the trigonometric pattern
    for j = 1:floor(Int, s/2)
        X[:,[2*j-1, 2*j]] = j == s/2 ? build_π(T, s, j) : build_std(T, s, j)
    end
    return X
end

function seasonal_regression(y::Vector{Fl}, s::Int, T::Int) where Fl
    X = build_X(s, T, Fl)
    return y - X * (X \ y)
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
              println("Didn't reject seasonal stationarity at 5% significance level"*
            " Test Statistic: $ω - Critical Value: $crit_val")
    return res > 0 ? false : true
end

