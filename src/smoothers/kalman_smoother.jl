function kalman_smoother!(
    smoother_output::SmootherOutput,
    system::Union{LinearUnivariateTimeInvariant,
                  LinearUnivariateTimeVariant,
                  LinearMultivariateTimeInvariant},
    filter_output::FilterOutput,
)
    multivariate_system = to_multivariate_time_variant(system)
    return kalman_smoother!(smoother_output, multivariate_system, filter_output)
end

function kalman_smoother!(
    smoother_output::SmootherOutput,
    system::LinearMultivariateTimeVariant{Fl},
    filter_output::FilterOutput,
) where Fl
    # Load dimensions data
    n = size(system.y, 1)
    m = size(system.T[1], 1)

    # Load filter data
    v = filter_output.v
    F = filter_output.F
    a = filter_output.a
    P = filter_output.P

    # Smoothed state and its covariance
    L = Vector{Matrix{Fl}}(undef, n)
    r = Vector{Vector{Fl}}(undef, n)
    N = Vector{Matrix{Fl}}(undef, n)

    # Initialization
    N[end] = zeros(Fl, m, m)
    r[end] = zeros(Fl, m)

    @inbounds for t in n:-1:2
        if any(y_ -> isnan(y_), system.y[t, :])
            r[t - 1] = system.T'[t] * r[t]
            N[t - 1] = system.T'[t] * N[t] * system.T[t]
        else
            Z_transp_invF = system.Z[t]' * inv(F[t])
            K = system.T[t] * P[t] * Z_transp_invF
            L[t] = system.T[t] - K * system.Z[t]
            r[t - 1] = Z_transp_invF * v[t] + L[t]' * r[t]
            N[t - 1] = Z_transp_invF * system.Z[t] + L[t]' * N[t] * L[t]
        end
        smoother_output.alpha[t] = a[t] + P[t] * r[t - 1]
        smoother_output.V[t] = P[t] - P[t] * N[t - 1] * P[t]
    end

    # Last iteration
    if any(y_ -> isnan(y_), system.y[1, :])
        r0 = system.T'[1] * r[1]
        N0 = system.T'[1] * N[1] * system.T[1]
    else
        Z_transp_invF = system.Z[1]' * inv(F[1])
        K = system.T[1] * P[1] * Z_transp_invF
        L[1] = system.T[1] - K * system.Z[1]
        r0 = Z_transp_invF * v[1] + L[1]' * r[1]
        N0 = Z_transp_invF * system.Z[1] + L[1]' * N[1] * L[1]
    end
    smoother_output.alpha[1] = a[1] + P[1] * r0
    smoother_output.V[1] = P[1] - P[1] * N0 * P[1]
    return smoother_output
end
