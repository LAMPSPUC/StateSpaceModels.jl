"""
    diagnostics(ss::StateSpace{T}) where T

Run diagnostics and print results
"""
function diagnostics(ss::StateSpace{T}; maxlag::Integer = 20) where T
    println("==============================================================")
    println("                    Running diagnostics...                    ")
    
    e = residuals(ss)
    pvalueJB = jarquebera(e)
    pvalueLB = ljungbox(e; maxlag = maxlag)
    pvalueH  = homoscedast(e)

    p = ss.model.dim.p
    for j = 1:p
        if p > 1
            println("--------------------------------------------------------------")
            println("                       Series $j of $p...                       ")
        end
        println("                Jarque-Bera: p-value = $(round(pvalueJB[j], digits = 5))                ")
        println("                  Ljung-Box: p-value = $(round(pvalueLB[j], digits = 5))                 ")
        println("           Homoscedasticity: p-value = $(round(pvalueH[j], digits = 5))              ")
    end

    println("==============================================================")

    return Diagnostics(pvalueJB, pvalueLB, pvalueH)
end

"""
    residuals(ss::StateSpace{T}) where T

Obtain standardized residuals from a state-space model
"""
function residuals(ss::StateSpace{T}) where T
    e = Matrix{T}(undef, ss.model.dim.n, ss.model.dim.p)
    for j = 1:ss.model.dim.p
        for t = 1:ss.model.dim.n
            e[t, j] = ss.filter.v[t, j] / sqrt(ss.filter.F[j, j, t])

            # Replace Inf with large numbers to avoid numerical issues
            e[t, j] == Inf && (e[t, j] = 1e6)
            e[t, j] == -Inf && (e[t, j] = -1e6)
        end
    end
    return e
end

"""
    jarquebera(e::Matrix{T}) where T

Run Jarque-Bera normality test and return p-values
"""
function jarquebera(e::Matrix{T}) where T
    n, p = size(e)
    pvalue = Vector{T}(undef, p)
    for j = 1:p
        m1 = sum(e[:, j])/n
        m2 = sum((e[:, j] .- m1).^2)/n
        m3 = sum((e[:, j] .- m1).^3)/n
        m4 = sum((e[:, j] .- m1).^4)/n
        b1 = (m3/m2^(3/2))^2
        b2 = (m4/m2^2)
        statistic = n * b1/6 + n*(b2 - 3)^2/24
        d = Chisq(2.)
        pvalue[j] = 1. - cdf(d, statistic)
    end
    return pvalue
end

"""
    ljungbox(e::Matrix{T}; maxlag::Int = 20) where T

Run Ljung-Box independence test and return p-values
"""
function ljungbox(e::Matrix{T}; maxlag::Int = 20) where T
    n, p = size(e)
    n <= maxlag && (maxlag = n-1)
    pvalue = Vector{T}(undef, p)
    for j = 1:p
        acor = autocor(e[:, j], 1:maxlag)
        Q = n*(n+2)*sum((acor[k]^2)/(n-k) for k = 1:maxlag)
        d = Chisq(maxlag)
        pvalue[j] = 1 - cdf(d, Q)
        isnan(pvalue[j]) && (pvalue[j] = 0.0)
    end
    return pvalue
end

"""
    homoscedast(e::Matrix{T}) where T

Run homoscedasticity test and return p-values
"""
function homoscedast(e::Matrix{T}) where T
    n, p = size(e)
    t1 = floor(Int, n/3)
    t2 = ceil(Int, 2*(n/3)+1)
    dist = FDist(n-t2+1, t1)
    pvalue = Vector{T}(undef, p)
    for j = 1:p
        H = sum(e[t, j]^2 for t = t2:n)/sum(e[t, j]^2 for t = 1:t1)
        Δ = abs(cdf(dist, H) - 0.5)
        H1 = quantile(dist, 0.5 - Δ)
        H2 = quantile(dist, 0.5 + Δ)
        pvalue[j] = cdf(dist, H1) + (1 - cdf(dist, H2))
    end
    return pvalue
end