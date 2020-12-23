# In total we can have trend, slope, seasonal, cycle and exogenous
# Maybe autoregressive
function validate_trend(trend::String)
    @assert trend in [
        "local level",
        "random walk",
        # "deterministic trend",
        # "linear deterministic trend",
        # "linear trend",
        "local linear trend",
        # "damped local lineaar trend",
        # "random walk with drift"
        "smooth trend"
    ]
    return true
end
function parse_trend(trend::String)
    validate_trend(trend)
    (has_irregular, has_trend, stochastic_trend, 
    has_slope, stochastic_slope) = (false, false, false, false, false)
    if trend == "local level"
        (has_irregular, has_trend, stochastic_trend, 
        has_slope, stochastic_slope) = (true, true, true, false, false)
    elseif trend == "random walk"
        (has_irregular, has_trend, stochastic_trend, 
        has_slope, stochastic_slope) = (false, true, true, false, false)
    elseif trend == "local linear trend"
        (has_irregular, has_trend, stochastic_trend, 
        has_slope, stochastic_slope) = (true, true, true, true, true)
    elseif trend == "smooth trend"
        (has_irregular, has_trend, stochastic_trend, 
        has_slope, stochastic_slope) = (true, true, false, true, true)
    end
    # Validate trend booleans
    if !has_trend && stochastic_trend
        error("Invalid trend specification.")
    end
    if !has_trend && has_slope
        error("Invalid trend specification.")
    end
    if !has_slope && stochastic_slope
        error("Invalid trend specification.")
    end
    return (has_irregular, has_trend, stochastic_trend, has_slope, stochastic_slope)
end
function validate_seasonal(seasonal::String)
    spl = split(seasonal)
    # TODO better error messaage
    # Maybe a no string can be also valid
    @assert length(spl) == 2
    @assert spl[1] in ["deterministic", "stochastic"]
    return true
end
function parse_seasonal(seasonal::String)
    validate_seasonal(seasonal)
    spl = split(seasonal)
    stochastic_seasonal = spl[1] == "stochastic"
    seasonal_freq = parse(Int, spl[2])
    has_seasonal = seasonal_freq == 0 ? false : true
    return has_seasonal, stochastic_seasonal, seasonal_freq
end

@doc raw"""
# TODO
- TREND

- SEASONAL


# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press.
"""
mutable struct UnobservedComponents <: StateSpaceModel
    hyperparameters::HyperParameters
    trend::String
    seasonal::String
    has_irregular::Bool
    has_trend::Bool
    stochastic_trend::Bool
    has_slope::Bool
    stochastic_slope::Bool
    # damped_slope::Bool TODO
    has_seasonal::Bool
    stochastic_seasonal::Bool
    # TODO
    # has_cycle
    # stochastic_cycle
    # damped_cycle
    seasonal_freq::Int
    system::LinearUnivariateTimeInvariant
    results::Results

    function UnobservedComponents(y::Vector{Fl}; 
                                  trend::String = "local level", # Multiple predefined options
                                  seasonal::String = "deterministic 0" # Stochastic or deterministic and period
                                #   cycle::String = "stochastic" # Stochastic, damped or deterministic
                                  ) where Fl

        (has_seasonal, stochastic_seasonal, seasonal_freq) = parse_seasonal(seasonal)
        (has_irregular, has_trend, stochastic_trend,
        has_slope, stochastic_slope) = parse_trend(trend)
        # Define system matrices
        Z = build_Z(Fl, has_trend, has_slope, has_seasonal, seasonal_freq)
        T = build_T(Fl, has_trend, has_slope, has_seasonal, seasonal_freq)
        R = build_R(Fl, has_trend, has_slope, has_seasonal, seasonal_freq,
                        stochastic_trend, stochastic_slope, stochastic_seasonal)
        d = zero(Fl)
        c = build_c(T)
        H = zero(Fl)
        Q = build_Q(R)

        system = LinearUnivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        # Define hyperparameters names
        names = build_names(has_irregular, stochastic_trend, stochastic_slope, stochastic_seasonal)
        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, trend, seasonal,
                    has_irregular, has_trend, stochastic_trend,
                    has_slope, stochastic_slope, has_seasonal,
                    stochastic_seasonal, seasonal_freq, system, Results{Fl}())
    end
end

function build_Z(Fl::DataType,
                 has_trend::Bool, 
                 has_slope::Bool, 
                 has_seasonal::Bool, 
                 seasonal_freq::Int)
    Z = Fl[]
    if has_trend
        Z = vcat(Z, one(Fl))
    end
    if has_slope
        Z = vcat(Z, zero(Fl))
    end
    if has_seasonal
        Z = vcat(Z, [1; zeros(Fl, seasonal_freq - 2)])
    end
    return Z
end
function build_T(Fl::DataType,
                 has_trend::Bool, 
                 has_slope::Bool, 
                 has_seasonal::Bool, 
                 seasonal_freq::Int)
    # Caalculate how maany states for each component
    num_trend_states = has_trend + has_slope
    num_seasonal_states = has_seasonal ? seasonal_freq - 1 : 0
    num_states = num_trend_states + num_seasonal_states
    T = zeros(Fl, num_states, num_states)
    if has_trend
        T[1, 1] = one(Fl)
    end
    if has_slope
        T[1, 2] = one(Fl)
        T[2, 2] = one(Fl)
    end
    if has_seasonal
        rows = num_trend_states + 1
        cols = num_trend_states + 1:num_trend_states + num_seasonal_states
        T[rows, cols] .= -one(Fl)
        rows = num_trend_states + 2:num_trend_states + num_seasonal_states
        cols = num_trend_states + 1:num_trend_states + num_seasonal_states - 1
        T[rows, cols] = Matrix{Fl}(I, seasonal_freq - 2, seasonal_freq - 2)
    end
    return T
end
function build_R(Fl::DataType,
                 has_trend::Bool, 
                 has_slope::Bool, 
                 has_seasonal::Bool, 
                 seasonal_freq::Int,
                 stochastic_trend::Bool, 
                 stochastic_slope::Bool, 
                 stochastic_seasonal::Bool)
    # Assign some model properties
    num_trend_states = has_trend + has_slope
    num_stochastic_trend = stochastic_trend + stochastic_slope
    num_seasonal_states = has_seasonal ? seasonal_freq - 1 : 0
    num_states = num_trend_states + num_seasonal_states
    num_stochastic_states = num_stochastic_trend + stochastic_seasonal

    R = zeros(num_states, num_stochastic_states)
    if stochastic_trend
        R[1, 1] = one(Fl)
    end
    if stochastic_slope
        if stochastic_trend
            R[2, 2] = one(Fl)
        else
            R[2, 1] = one(Fl)
        end
    end
    if stochastic_seasonal
        R[num_stochastic_trend + 1, num_stochastic_trend + 1] = one(Fl)
    end
    return R
end
function build_c(T::Matrix{Fl}) where Fl
    m = size(T, 1)
    return zeros(Fl, m)
end
function build_Q(R::Matrix{Fl}) where Fl
    r = size(R, 2)
    return Matrix{Fl}(I, r, r)
end
function build_names(has_irregular::Bool,
                     stochastic_trend::Bool, 
                     stochastic_slope::Bool, 
                     stochastic_seasonal::Bool)
    names = String[]
    if has_irregular
        push!(names, "sigma2_irregular")
    end
    if stochastic_trend
        push!(names, "sigma2_trend")
    end
    if stochastic_slope
        push!(names, "sigma2_slope")
    end
    if stochastic_seasonal
        push!(names, "sigma2_seasonal")
    end
    return names
end

has_sigma2(str::String) = occursin("sigma2", str)
function sigma2_names(model::UnobservedComponents)
    return filter(has_sigma2, get_names(model))
end

# Obligatory methods
function default_filter(model::UnobservedComponents)
    Fl = typeof_model_elements(model)
    steadystate_tol = Fl(1e-5)
    a1 = zeros(Fl, num_states(model))
    P1 = Fl(1e6) .* Matrix{Fl}(I, num_states(model), num_states(model))
    return UnivariateKalmanFilter(a1, P1, num_states(model), steadystate_tol)
end

function initial_hyperparameters!(model::UnobservedComponents)
    Fl = typeof_model_elements(model)
    # TODO add heuristic for initial hyperparameters
    initial_hyperparameters = Dict{String,Fl}(get_names(model) .=> one(Fl))
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::UnobservedComponents)
    for variance_name in sigma2_names(model)
        constrain_variance!(model, variance_name)
    end
    return model
end

function unconstrain_hyperparameters!(model::UnobservedComponents)
    for variance_name in sigma2_names(model)
        unconstrain_variance!(model, variance_name)
    end
    return model
end

function fill_model_system!(model::UnobservedComponents)
    num_trend_states = model.has_trend + model.has_slope
    num_stochastic_trend = model.stochastic_trend + model.stochastic_slope
    # num_seasonal_states = has_seasonal ? seasonal_freq - 1 : 0
    # num_states = num_trend_states + num_seasonal_states
    # num_stochastic_states = num_stochastic_trend + stochastic_seasonal

    if model.has_irregular
        model.system.H = get_constrained_value(model, "sigma2_irregular")
    end
    if model.stochastic_trend
        model.system.Q[1, 1] = get_constrained_value(model, "sigma2_trend")
    end
    if model.stochastic_slope
        if model.stochastic_trend
            model.system.Q[2, 2] = get_constrained_value(model, "sigma2_slope")
        else
            model.system.Q[1, 1] = get_constrained_value(model, "sigma2_slope")
        end
    end
    if model.stochastic_seasonal
        model.system.Q[num_stochastic_trend + 1, num_stochastic_trend + 1] = get_constrained_value(model, "sigma2_seasonal")
    end

    return model
end

function reinstantiate(model::UnobservedComponents, y::Vector{Fl}) where Fl
    return UnobservedComponents(y; trend = model.trend, seasonal = model.seasonal)
end

has_exogenous(::UnobservedComponents) = false
