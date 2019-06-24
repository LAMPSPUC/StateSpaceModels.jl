"""
    sqrt_kalman_filter(model::StateSpaceModel, sqrtH::Matrix{Typ}, sqrtQ::Matrix{Typ}; tol::Float64 = 1e-5) where Typ <: AbstractFloat

Square-root Kalman filter with big Kappa initialization.
"""
function sqrt_kalman_filter(model::StateSpaceModel, sqrtH::Matrix{Typ}, sqrtQ::Matrix{Typ}; tol::Typ = 1e-5) where Typ <: AbstractFloat

    # Load dimensions
    n, p, m, r = size(model)

    # Load system
    y = model.y
    Z, T, R = ztr(model)

    # Predictive state and its sqrt-covariance
    a     = Matrix{Float64}(undef, n+1, m)
    sqrtP = Array{Float64, 3}(undef, m, m, n+1)

    # Innovation and its sqrt-covariance
    v     = Matrix{Float64}(undef, n, p)
    sqrtF = Array{Float64, 3}(undef, p, p, n)

    # Kalman gain
    K     = Array{Float64, 3}(undef, m, p, n)

    # Auxiliary matrices
    U2star = Array{Float64, 3}(undef, m, p, n)
    
    # Steady state initialization
    steadystate  = false
    tsteady      = n+1

    # Initial state: big Kappa initialization
    a[1, :]        = zeros(m, 1)
    sqrtP[:, :, 1]    = 1e6 .* Matrix(I, m, m)

    # Pre-allocating for performance
    zeros_pr = zeros(p, r)
    zeros_mp = zeros(m, p)
    range1   = (p + 1):(p + m)
    range2   = 1:p
    sqrtH_zeros_pr  = [sqrtH zeros_pr]
    zeros_mp_RsqrtQ = [zeros_mp R*sqrtQ]

    # Square-root Kalman filter
    for t = 1:n
        v[t, :] = y[t, :] - Z[:, :, t]*a[t, :]
        if steadystate
            sqrtF[:, :, t] = sqrtF[:, :, t-1]
            K[:, :, t] = K[:, :, t-1]
            a[t+1, :] = T * a[t, :] + K[:, :, t] * v[t, :]
            sqrtP[:, :, t+1] = sqrtP[:, :, t]
        else
            # Manipulation of auxiliary matrices
            U         = SMatrix{p + m, m + p + r}([Z[:, :, t]*sqrtP[:, :, t] sqrtH_zeros_pr;
                                                   T*sqrtP[:, :, t]         zeros_mp_RsqrtQ])
            G         = qr(U').Q
            Ustar     = U*G
            U2star[:, :, t] = Ustar[range1, range2]
            sqrtF[:, :, t]  = Ustar[range2, range2]

            # Kalman gain and predictive state update
            K[:, :, t]       = U2star[:, :, t]*inv(sqrtF[:, :, t])
            a[t+1, :]        = T*a[t, :] + K[:, :, t]*v[t, :]
            sqrtP[:, :, t+1] = Ustar[range1, range1]

            # Checking if steady state was attained
            if check_steady_state(sqrtP[:, :, t+1], sqrtP[:, :, t], tol)
                steadystate = true
                tsteady     = t
            end
        end
    end

    # Return the auxiliary filter structre
    return SquareRootFilter(a[1:end-1, :], v, sqrtP, sqrtF, steadystate, tsteady, K)
end

"""
    sqrt_smoother(model::StateSpaceModel, sqrt_filter::SquareRootFilter)

Square-root smoother for state space model.
"""
function sqrt_smoother(model::StateSpaceModel, sqrt_filter::SquareRootFilter)

    # Load dimensions data
    n, p, m, r = size(model)

    # Load system data
    Z, T, R = ztr(model)

    # Load filter data
    a           = sqrt_filter.a
    v           = sqrt_filter.v
    tsteady     = sqrt_filter.tsteady
    sqrtF       = sqrt_filter.sqrtF
    sqrtP       = sqrt_filter.sqrtP
    K           = sqrt_filter.K

    # Smoothed state and its covariance
    alpha = Matrix{Float64}(undef, n, m)
    V     = Array{Float64, 3}(undef, m, m, n)
    L     = Array{Float64, 3}(undef, m, m, n)
    r     = Matrix{Float64}(undef, n, m)
    sqrtN = Array{Float64, 3}(undef, m, m, n)

    # Initialization
    sqrtN[:, :, end]  = zeros(m, m)
    r[end, :]      = zeros(m, 1)
    sqrtPsteady = sqrtP[:, :, end]
    sqrtFsteady = sqrtF[:, :, end]

    # Iterating backwards
    for t = n:-1:tsteady
        Psteady = gram(sqrtP[:, :, end])
        Fsteady = gram(sqrtF[:, :, end])
        N_t     = gram(sqrtN[:, :, t])
        L[:, :, t]   = T - K[:, :, end] * Z[:, :, t]
        r[t-1, :] = Z[:, :, t]' * inv(Fsteady) * v[t, :] + L[:, :, t]' * r[t, :]

        # QR decomposition of auxiliary matrix Nstar
        Nstar        = [Z[:, :, t]' * inv(sqrtF[:, :, end]) L[:, :, t]' * sqrtN[:, :, t]]
        G            = qr(Nstar').Q
        NstarG       = Nstar * G
        sqrtN[:, :, t-1]   = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        alpha[t, :] = a[t, :] + Psteady * r[t-1, :]
        V[:, :, t]  = Psteady - Psteady * N_t * Psteady
    end

    for t = tsteady-1:-1:2
        L[:, :, t]   = T - K[:, :, t] * Z[:, :, t]
        r[t-1, :] = Z[:, :, t]' * inv(sqrtF[:, :, t] * sqrtF[:, :, t]') * v[t, :] + L[:, :, t]'*r[t, :]
        Nstar  = [Z[:, :, t]' * inv(sqrtF[:, :, t]) L[:, :, t]' * sqrtN[:, :, t]]

        # QR decomposition of auxiliary matrix Nstar
        NstarG     = Nstar*qr(Nstar').Q
        sqrtN[:, :, t-1] = NstarG[1:m, 1:m]

        # Smoothed state and its covariance
        P = gram(sqrtP[:, :, t])
        N = gram(sqrtN[:, :, t])
        alpha[t, :] = a[t, :] + P * r[t-1, :]
        V[:, :, t]  = P - (P * N * P)
    end

    L[:, :, 1]   = T - K[:, :, 1] * Z[:, :, 1]
    r_0    = Z[:, :, 1]' * inv(gram(sqrtF[:, :, 1])) * v[1, :] + L[:, :, 1]' * r[1, :]
    Nstar  = [Z[:, :, 1]' * inv(sqrtF[:, :, 1]) L[:, :, 1]' * sqrtN[:, :, 1]]
    G      = qr(Nstar').Q
    NstarG = Nstar*G
    
    sqrtN_0  = NstarG[1:m, 1:m]
    P_1 = gram(sqrtP[:, :, 1])
    alpha[1, :] = a[1, :] + P_1 * r_0
    V[:, :, 1]  = P_1 - (P_1 * gram(sqrtN_0) * P_1)

    # Return the Square Root kalman filter smoothed state
    return Smoother(alpha, V)
end


# All filters have to have implemented the following functions
# *
# *
# *

function statespace_covariance(psi::Vector{T}, p::Int, r::Int,
                               filter_type::Type{SquareRootFilter}) where T <: AbstractFloat
    # Observation sqrt-covariance matrix
    if p > 1
        sqrtH     = tril!(ones(p, p))
        unknownsH = Int(p*(p + 1)/2)
        sqrtH[findall(isequal(1), sqrtH)] = psi[1:unknownsH]
    else
        sqrtH = psi[1].*ones(1, 1)
        unknownsH = 1
    end

    # State sqrt-covariance matrix
    sqrtQ = kron(Matrix{Float64}(I, Int(r/p), Int(r/p)), tril!(ones(p, p)))
    sqrtQ[findall(x -> x == 1, sqrtQ)] = psi[(unknownsH+1):Int(unknownsH + (r/p)*(p*(p + 1)/2))]

    return sqrtH, sqrtQ
end

function get_log_likelihood_params(psitilde::Vector{T}, model::StateSpaceModel, 
                                   filter_type::Type{SquareRootFilter}) where T <: AbstractFloat

    sqrtH, sqrtQ = statespace_covariance(psitilde, model.dim.p, model.dim.r, filter_type)
    # Obtain innovation v and its variance F
    sqrt_kfilter = sqrt_kalman_filter(model, sqrtH, sqrtQ)
    # Return v and F
    return sqrt_kfilter.v, gram_in_time(sqrt_kfilter.sqrtF)
end

function kalman_filter_and_smoother(model::StateSpaceModel, covariance::StateSpaceCovariance, 
                                    filter_type::Type{SquareRootFilter})
    # Compute sqrt matrices                                
    sqrtH = cholesky(covariance.H).L # .L stands for Lower triangular
    sqrtQ = cholesky(covariance.Q).L # .L stands for Lower triangular

    # Do the SquareRootFilter 
    filtered_state = sqrt_kalman_filter(model, sqrtH.data, sqrtQ.data)
    smoothed_state = sqrt_smoother(model, filtered_state)
    return FilteredState(filtered_state.a, filtered_state.v, 
                         gram_in_time(filtered_state.sqrtP), gram_in_time(filtered_state.sqrtF),
                         filtered_state.steadystate, filtered_state.tsteady) ,
           SmoothedState(smoothed_state.alpha, smoothed_state.V) 
end