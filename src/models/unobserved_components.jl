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
    if seasonal == "no"
        return true
    else
        spl = split(seasonal)
        # TODO better error messaage
        # Maybe a no string can be also valid
        @assert length(spl) == 2
        @assert spl[1] in ["deterministic", "stochastic"]
        return true
    end
end
function parse_seasonal(seasonal::String)
    validate_seasonal(seasonal)
    if seasonal == "no"
        has_seasonal, stochastic_seasonal, seasonal_freq = (false, false, 0)
        return has_seasonal, stochastic_seasonal, seasonal_freq
    else
        spl = split(seasonal)
        stochastic_seasonal = spl[1] == "stochastic"
        seasonal_freq = parse(Int, spl[2])
        has_seasonal = seasonal_freq == 0 ? false : true
        return has_seasonal, stochastic_seasonal, seasonal_freq
    end
end
function validate_cycle(cycle::String)
    if cycle == "no"
        return true
    else
        spl = split(cycle)
        # TODO better error messaage
        if length(spl) == 1
            @assert spl[1] in ["deterministic", "stochastic"]
            return true
        elseif length(spl) == 2
            @assert spl[1] in ["deterministic", "stochastic"]
            @assert spl[2] == "damped"
            return true
        end
    end
    return false
end
function parse_cycle(cycle::String)
    validate_cycle(cycle)
    if cycle == "no"
        has_cycle, stochastic_cycle, damped_cycle = (false, false, false)
        return has_cycle, stochastic_cycle, damped_cycle
    else
        spl = split(cycle)
        has_cycle = true
        # TODO better error messaage
        # Maybe a no string can be also valid
        if length(spl) == 1
            stochastic_cycle = spl[1] == "stochastic"
            return has_cycle, stochastic_cycle, false
        elseif length(spl) == 2
            stochastic_cycle = spl[1] == "stochastic"
            damped_cycle = spl[2] == "damped"
            return has_cycle, stochastic_cycle, damped_cycle
        end
    end
end

@doc raw"""
    UnobservedComponents(
        y::Vector{Fl}; 
        trend::String = "local level",
        seasonal::String = "no"
        cycle::String = "no"
    ) where Fl

An unobserved components model that can have trend/level, seasonal and cycle components. 
Each component should be specified by strings, if the component is not desired in the model 
a string with "no" can be passed as keyword argument. 
    
These models take the general form 

```math
\begin{gather*}
    \begin{aligned}
    y_t = \mu_t + \gamma_t + c_t + \varepsilon_t
    \end{aligned}
\end{gather*}
```
where ``y_t`` refers to the observation vector at time ``t``,
``\mu_t`` refers to the trend component, ``\gamma_t`` refers to the
seasonal component, ``c_t`` refers to the cycle, and
``\varepsilon_t`` is the irregular. The modeling details of these
components are given below.

### **Trend**

The trend component can be modeled in a lot of different ways, usually it is called level when
there is no slope component. The modelling options can be expressed as in the example `trend = "local level"`.

* Local Level
string: `"local level"`
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \varepsilon_{t} \quad \varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \eta_{t} \quad \eta_{t} \sim \mathcal{N}(0, \sigma^2_{\eta})\\
    \end{aligned}
\end{gather*}
```

* Random Walk
string: `"random walk"`
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t}\\
        \mu_{t+1} &= \mu_{t} + \eta_{t} \quad \eta_{t} \sim \mathcal{N}(0, \sigma^2_{\eta})\\
    \end{aligned}
\end{gather*}
```

* Local Linear Trend
string: `"local linear trend"`
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \gamma_{t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \nu_{t} + \xi_{t} \quad &\xi_{t} \sim \mathcal{N}(0, \sigma^2_{\xi})\\
        \nu_{t+1} &= \nu_{t} + \zeta_{t} \quad &\zeta_{t} \sim \mathcal{N}(0, \sigma^2_{\zeta})\\
    \end{aligned}
\end{gather*}
```

* Smooth Trend
string: `"smooth trend"`
```math
\begin{gather*}
    \begin{aligned}
        y_{t} &=  \mu_{t} + \gamma_{t} + \varepsilon_{t} \quad &\varepsilon_{t} \sim \mathcal{N}(0, \sigma^2_{\varepsilon})\\
        \mu_{t+1} &= \mu_{t} + \nu_{t}\\
        \nu_{t+1} &= \nu_{t} + \zeta_{t} \quad &\zeta_{t} \sim \mathcal{N}(0, \sigma^2_{\zeta})\\
    \end{aligned}
\end{gather*}
```

**Seasonal**

The seasonal component is modeled as:

```math
\begin{gather*}
    \begin{aligned}
    \gamma_t = - \sum_{j=1}^{s-1} \gamma_{t+1-j} + \omega_t \quad \omega_t \sim N(0, \sigma^2_\omega)
    \end{aligned}
\end{gather*}
```

The periodicity (number of seasons) is s, and the defining character is
that (without the error term), the seasonal components sum to zero across
one complete cycle. The inclusion of an error term allows the seasonal
effects to vary over time. The modelling options can be expressed in terms
of `"deterministic"` or `"stochastic"` and the periodicity as a number in 
the string, i.e., `seasonal = "stochastic 12"`.

**Cycle**

The cycle component is modeled as

```math
\begin{gather*}
    \begin{aligned}
        c_{t+1} &= \rho_c \left(c_{t} \cos(\lambda_c) + c_{t}^{*} \sin(\lambda_c)\right) \quad & \tilde\omega_{t} \sim \mathcal{N}(0, \sigma^2_{\tilde\omega})\\
        c_{t+1}^{*} &= \rho_c \left(-c_{t} \sin(\lambda_c) + c_{t}^{*} \sin(\lambda_c)\right) \quad &\tilde\omega^*_{t} \sim \mathcal{N}(0, \sigma^2_{\tilde\omega})\\
    \end{aligned}
\end{gather*}
```

The cyclical component is intended to capture cyclical effects at time frames much longer 
than captured by the seasonal component. The parameter ``\lambda_c`` is the frequency of the cycle
and it is estimated via maximum likelihood. The inclusion of error terms allows the cycle
effects to vary over time. The modelling options can be expressed in terms
of `"deterministic"` or `"stochastic"` and the damping effect as a string, i.e., 
`cycle = "stochastic"`, `cycle = "deterministic"` or `cycle = "damped"`.

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press.
"""
mutable struct UnobservedComponents <: StateSpaceModel
    hyperparameters::HyperParameters
    trend::String
    seasonal::String
    cycle::String
    has_irregular::Bool
    has_trend::Bool
    stochastic_trend::Bool
    has_slope::Bool
    stochastic_slope::Bool
    # damped_slope::Bool TODO
    has_seasonal::Bool
    stochastic_seasonal::Bool
    seasonal_freq::Int
    has_cycle::Bool
    stochastic_cycle::Bool
    damped_cycle::Bool
    system::LinearUnivariateTimeInvariant
    results::Results

    function UnobservedComponents(y::Vector{Fl}; 
                                  trend::String = "local level",
                                  seasonal::String = "no", # for example "deterministic 3" or "stochastic 12"
                                  cycle::String = "no" # for example "deterministic damped" or "stochastic"
                                  ) where Fl

        (has_irregular, has_trend, stochastic_trend,
        has_slope, stochastic_slope) = parse_trend(trend)
        (has_seasonal, stochastic_seasonal, seasonal_freq) = parse_seasonal(seasonal)
        (has_cycle, stochastic_cycle, damped_cycle) = parse_cycle(cycle)
        # Define system matrices
        Z = build_Z(Fl, has_trend, has_slope, has_seasonal, seasonal_freq, has_cycle)
        T = build_T(Fl, has_trend, has_slope, has_seasonal, seasonal_freq, has_cycle)
        R = build_R(Fl, has_trend, has_slope, has_seasonal, seasonal_freq, has_cycle,
                        stochastic_trend, stochastic_slope, stochastic_seasonal, stochastic_cycle)
        d = zero(Fl)
        c = build_c(T)
        H = zero(Fl)
        Q = build_Q(R)

        system = LinearUnivariateTimeInvariant{Fl}(y, Z, T, R, d, c, H, Q)

        # Define hyperparameters names
        names = build_names(has_irregular, stochastic_trend, stochastic_slope, 
                            stochastic_seasonal, has_cycle, stochastic_cycle, damped_cycle)
        hyperparameters = HyperParameters{Fl}(names)

        return new(hyperparameters, trend, seasonal, cycle,
                    has_irregular, has_trend, stochastic_trend,
                    has_slope, stochastic_slope, has_seasonal,
                    stochastic_seasonal, seasonal_freq, 
                    has_cycle, stochastic_cycle, damped_cycle,
                    system, Results{Fl}())
    end
end

function build_Z(Fl::DataType,
                 has_trend::Bool, 
                 has_slope::Bool, 
                 has_seasonal::Bool, 
                 seasonal_freq::Int,
                 has_cycle::Bool)
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
    if has_cycle
        Z = vcat(Z, [one(Fl); zero(Fl)])
    end
    return Z
end
function build_T(Fl::DataType,
                 has_trend::Bool, 
                 has_slope::Bool, 
                 has_seasonal::Bool, 
                 seasonal_freq::Int,
                 has_cycle::Bool)
    # Caalculate how maany states for each component
    num_trend_states = has_trend + has_slope
    num_seasonal_states = has_seasonal ? seasonal_freq - 1 : 0
    num_cycle_states = 2 * has_cycle
    num_states = num_trend_states + num_seasonal_states + num_cycle_states
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
    if has_cycle
        # Do nothing T should be filled with the cycle expression
    end
    return T
end
function build_R(Fl::DataType,
                 has_trend::Bool, 
                 has_slope::Bool, 
                 has_seasonal::Bool, 
                 seasonal_freq::Int,
                 has_cycle::Bool,
                 stochastic_trend::Bool, 
                 stochastic_slope::Bool, 
                 stochastic_seasonal::Bool,
                 stochastic_cycle::Bool)
    # Assign some model properties
    num_trend_states = has_trend + has_slope
    num_stochastic_trend = stochastic_trend + stochastic_slope
    num_seasonal_states = has_seasonal ? seasonal_freq - 1 : 0
    num_cycle_states = 2 * has_cycle
    num_states = num_trend_states + num_seasonal_states + num_cycle_states
    num_stochastic_states = num_stochastic_trend + stochastic_seasonal + 2 * stochastic_cycle

    R = zeros(num_states, num_stochastic_states)
    for i in 1:num_stochastic_states
        R[i, i] = one(Fl)
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
                     stochastic_seasonal::Bool,
                     has_cycle::Bool, 
                     stochastic_cycle::Bool, 
                     damped_cycle::Bool)
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
    if has_cycle
        push!(names, "λ_cycle")
        if stochastic_cycle
            push!(names, "sigma2_cycle")
        end
        if damped_cycle
            push!(names, "ρ_cycle")
        end
    end
    return names
end

has_sigma2(str::String) = occursin("sigma2", str)
function sigma2_names(model::UnobservedComponents)
    return filter(has_sigma2, get_names(model))
end

function diag_indices(k::Int)
    idx = Int[]
    l = 1
    for i in 1:k, j in 1:k
        if i == j
            push!(idx, l)
        end
        l += 1
    end
    return idx
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
    y = filter(!isnan, model.system.y)
    observed_variance = var(y)
    # TODO add heuristic for initial hyperparameters
    initial_hyperparameters = Dict{String,Fl}(get_names(model) .=> one(Fl))
    if model.has_irregular
        initial_hyperparameters["sigma2_irregular"] = observed_variance
    end
    if model.has_trend
        initial_hyperparameters["sigma2_trend"] = observed_variance
    end
    if model.has_cycle
        initial_hyperparameters["λ_cycle"] = Fl(2 * pi / 12)
        if model.damped_cycle
            initial_hyperparameters["ρ_cycle"] = Fl(0.7)
        end
    end
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return model
end

function constrain_hyperparameters!(model::UnobservedComponents)
    Fl = typeof_model_elements(model)
    for variance_name in sigma2_names(model)
        constrain_variance!(model, variance_name)
    end
    if model.has_cycle
        # Durbin and Koopman (2012) comment possible values in their book pp. 48
        constrain_box!(model, "λ_cycle", Fl(2 * pi / 100), Fl(2 * pi / 1.5))
        if model.damped_cycle
            constrain_box!(model, "ρ_cycle", zero(Fl), Fl(1.2))
        end
    end
    return model
end

function unconstrain_hyperparameters!(model::UnobservedComponents)
    Fl = typeof_model_elements(model)
    for variance_name in sigma2_names(model)
        unconstrain_variance!(model, variance_name)
    end
    if model.has_cycle
        # Durbin and Koopman (2012) comment possible values in their book pp. 48
        unconstrain_box!(model, "λ_cycle", Fl(2 * pi / 100), Fl(2 * pi / 1.5))
        if model.damped_cycle
            unconstrain_box!(model, "ρ_cycle", zero(Fl), Fl(1.2))
        end
    end
    return model
end

function fill_model_system!(model::UnobservedComponents)
    # TODO (performance) maybe cache this indices
    # This creates some extra allocations
    diag_Q_idx = diag_indices(size(model.system.Q, 1))
    idx = 1

    if model.has_irregular
        model.system.H = get_constrained_value(model, "sigma2_irregular")
    end
    if model.stochastic_trend
        model.system.Q[diag_Q_idx[idx]] = get_constrained_value(model, "sigma2_trend")
        idx += 1
    end
    if model.stochastic_slope
        model.system.Q[diag_Q_idx[idx]] = get_constrained_value(model, "sigma2_slope")
        idx += 1
    end
    if model.stochastic_seasonal
        model.system.Q[diag_Q_idx[idx]] = get_constrained_value(model, "sigma2_seasonal")
        idx += 1
    end
    if model.stochastic_cycle
        model.system.Q[diag_Q_idx[idx]] = get_constrained_value(model, "sigma2_cycle")
        idx += 1
        model.system.Q[diag_Q_idx[idx]] = get_constrained_value(model, "sigma2_cycle")
        idx += 1
    end

    if model.has_cycle
        num_trend_states = model.has_trend + model.has_slope
        num_seasonal_states = model.has_seasonal ? model.seasonal_freq - 1 : 0
        num_trend_seasonal_states = num_trend_states + num_seasonal_states
        cycle_rows = num_trend_seasonal_states+1:num_trend_seasonal_states+2
        cycle_cols = cycle_rows
        if model.damped_cycle
            ρ = get_constrained_value(model, "ρ_cycle")
            model.system.T[cycle_rows[1], cycle_rows[1]] = ρ * cos(get_constrained_value(model, "λ_cycle"))
            model.system.T[cycle_rows[2], cycle_rows[1]] = ρ * -sin(get_constrained_value(model, "λ_cycle"))
            model.system.T[cycle_rows[1], cycle_rows[2]] = ρ * sin(get_constrained_value(model, "λ_cycle"))
            model.system.T[cycle_rows[2], cycle_rows[2]] = ρ * cos(get_constrained_value(model, "λ_cycle"))
        else
            model.system.T[cycle_rows[1], cycle_rows[1]] = cos(get_constrained_value(model, "λ_cycle"))
            model.system.T[cycle_rows[2], cycle_rows[1]] = -sin(get_constrained_value(model, "λ_cycle"))
            model.system.T[cycle_rows[1], cycle_rows[2]] = sin(get_constrained_value(model, "λ_cycle"))
            model.system.T[cycle_rows[2], cycle_rows[2]] = cos(get_constrained_value(model, "λ_cycle"))
        end
    end

    return model
end

function reinstantiate(model::UnobservedComponents, y::Vector{Fl}) where Fl
    return UnobservedComponents(y; 
                                trend = model.trend, 
                                seasonal = model.seasonal,
                                cycle = model.cycle)
end

has_exogenous(::UnobservedComponents) = false
