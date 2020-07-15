export LinearRegression

struct RegressionHyperParametersAuxiliary
    beta_names::Vector{String}
    function RegressionHyperParametersAuxiliary(num_states::Int)
        return new(beta_names(num_states))
    end
end

struct LinearRegression{Fl <: Real} <: StateSpaceModel
    hyperparameters::HyperParameters{Fl}
    hyperparameters_auxiliary::RegressionHyperParametersAuxiliary
    system::LinearUnivariateTimeVariant{Fl}

    function LinearRegression(X::Matrix{Fl}, y::Vector{Fl}) where Fl

        # Treat possible input errors
        @assert length(y) == size(X, 1)

        num_observations = size(X, 1)
        num_exogenous = size(X, 2)

        Z = [X[t, :] for t in 1:num_observations]
        T = [Matrix{Fl}(I, num_exogenous, num_exogenous) for _ in 1:num_observations]
        R = [zeros(num_exogenous, 1) for _ in 1:num_observations]
        d = [zero(Fl) for _ in 1:num_observations]
        c = [zeros(num_exogenous) for _ in 1:num_observations]
        H = [one(Fl) for _ in 1:num_observations]
        Q = [zeros(Fl, 1, 1) for _ in 1:num_observations]

        system = LinearUnivariateTimeVariant{Fl}(y, Z, T, R, d, c, H, Q)

        names = [["β_$i" for i in 1:num_exogenous]; "sigma2_ε"]
        hyperparameters = HyperParameters{Fl}(names)

        hyperparameters_auxiliary = RegressionHyperParametersAuxiliary(get_num_states(system))

        return new{Fl}(hyperparameters, hyperparameters_auxiliary, system)
    end
end

function beta_names(num_states::Int)
    str = Vector{String}(undef, num_states)
    for i in eachindex(str)
        str[i] = "β_$i"
    end
    return str
end
get_beta_name(model::LinearRegression, i::Int) = model.hyperparameters_auxiliary.beta_names[i]
fill_H_in_time(model::LinearRegression, H::Fl) where Fl = fill_system_matrice_with_value_in_time(model.system.H, H)

# Obligatory functions
function default_filter(model::LinearRegression{Fl}) where Fl
    a1 = zeros(Fl, get_num_states(model))
    return RegressionKalmanFilter(a1)
end
function initial_hyperparameters!(model::LinearRegression{Fl}) where Fl
    initial_hyperparameters = Dict{String, Float64}(
        "sigma2_ε" => var(model.system.y)
    )
    # The optimal regressors are the result of X \ y
    betas = hcat(model.system.Z...)' \ model.system.y
    for i in 1:get_num_states(model)
        initial_hyperparameters[get_beta_name(model, i)] = betas[i]
    end
    set_initial_hyperparameters!(model, initial_hyperparameters)
    return
end
function constraint_hyperparameters!(model::LinearRegression{Fl}) where {Fl}
    for i in 1:get_num_states(model)
        constrain_identity(model, get_beta_name(model, i))
    end
    constrain_variance(model, "sigma2_ε")
    return
end
function unconstraint_hyperparameters!(model::LinearRegression{Fl}) where Fl
    for i in 1:get_num_states(model)
        unconstrain_identity(model, get_beta_name(model, i))
    end
    unconstrain_variance(model, "sigma2_ε")
    return
end
function update!(model::LinearRegression{Fl}) where Fl
    # Fill the same H for every timestamp
    H = get_constrained_value(model, "sigma2_ε")
    fill_H_in_time(model, H)
    return 
end
function update!(filter::KalmanFilter, model::LinearRegression{Fl}) where Fl
    for i in axes(filter.kalman_state.a, 1)
        filter.kalman_state.a[i] = get_constrained_value(model, get_beta_name(model, i))
    end
end