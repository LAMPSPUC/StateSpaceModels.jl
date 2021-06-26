@doc raw"""
    fit!(
        model::StateSpaceModel;
        filter::KalmanFilter=default_filter(model),
        optimizer::Optimizer=Optimizer(Optim.LBFGS())
    )

Estimate the state-space model parameters via maximum likelihood. The resulting optimal
hyperparameters and the corresponding log-likelihood are stored within the model. You can
choose the desired filter method (`UnivariateKalmanFilter`, `ScalarKalmanFilter`, etc.) and
the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) optimization algortihm. 

# Example
```jldoctest
julia> model = LocalLevel(rand(100))
LocalLevel model

julia> fit!(model)
LocalLevel model

julia> model = LocalLinearTrend(LinRange(1, 100, 100) + rand(100))
LocalLinearTrend model

julia> fit!(model; optimizer = Optimizer(StateSpaceModels.Optim.NelderMead()))
LocalLinearTrend model
```
"""
function fit!(
    model::StateSpaceModel;
    filter::KalmanFilter=default_filter(model),
    optimizer::Optimizer=default_optimizer(model),
)
    isfitted(model) && return model
    @assert has_fit_methods(typeof(model))
    initial_unconstrained_hyperparameter = handle_optim_initial_hyperparameters(model)
    # TODO Should there be a try catch?
    func = TwiceDifferentiable(
        x -> -optim_loglike(model, filter, x), initial_unconstrained_hyperparameter
    )
    opt = optimize(
        func, initial_unconstrained_hyperparameter, optimizer.method, optimizer.options
    )
    # optim_loglike returns the mean loglike for numerical stability purposes
    # the optimum loglike must be rescaled
    opt_loglikelihood = -opt.minimum * size(model.system.y, 1)
    opt_hyperparameters = opt.minimizer
    update_model_hyperparameters!(model, opt_hyperparameters)
    # TODO
    # I leaned that this is not a good way to compute the covariance matrix of parameters
    # we should investigate other methods
    numerical_hessian = Optim.hessian!(func, opt_hyperparameters)
    std_err = diag(pinv(numerical_hessian))
    fill_results!(model, opt_loglikelihood, std_err)
    return model
end

"""
    has_fit_methods(model_type::Type{<:StateSpaceModel}) -> Bool

Verify if a certain `StateSpaceModel` has the necessary methods to perform `fit!``.
"""
function has_fit_methods(model_type::Type{<:StateSpaceModel})
    tuple_with_model_type = Tuple{model_type}
    m1 = hasmethod(default_filter, tuple_with_model_type)
    m2 = hasmethod(initial_hyperparameters!, tuple_with_model_type)
    m3 = hasmethod(constrain_hyperparameters!, tuple_with_model_type)
    m4 = hasmethod(unconstrain_hyperparameters!, tuple_with_model_type)
    m5 = hasmethod(fill_model_system!, tuple_with_model_type)
    return m1 && m2 && m3 && m4 && m5
end

struct CoefficientTable{Fl<:AbstractFloat}
    names::Vector{String}
    coef::Vector{Fl}
    std_err::Vector{Fl}
    z::Vector{Fl}
    p_value::Vector{Fl}

    function CoefficientTable{Fl}(
        names::Vector{String},
        coef::Vector{Fl},
        std_err::Vector{Fl},
        z::Vector{Fl},
        p_value::Vector{Fl},
    ) where Fl
        @assert length(names) ==
                length(coef) ==
                length(std_err) ==
                length(z) ==
                length(p_value)
        return new{Fl}(names, coef, std_err, z, p_value)
    end
end

function CoefficientTable{Fl}() where Fl
    return CoefficientTable{Fl}(String[], Fl[], Fl[], Fl[], Fl[])
end

Base.length(coef_table::CoefficientTable) = length(coef_table.names)

function Base.isempty(coef_table::CoefficientTable)
    return isempty(coef_table.names) &&
           isempty(coef_table.coef) &&
           isempty(coef_table.std_err) &&
           isempty(coef_table.z) &&
           isempty(coef_table.p_value)
end

mutable struct Results{Fl<:AbstractFloat}
    model_name::String
    coef_table::CoefficientTable{Fl}
    llk::Fl
    aic::Fl
    aicc::Fl
    bic::Fl
    num_observations::Int
    num_hyperparameters::Int
end

function Results{Fl}() where Fl
    return Results{Fl}("", CoefficientTable{Fl}(), Fl(NaN), Fl(NaN), Fl(NaN), Fl(NaN), 0, 0)
end

"""
    results(model::StateSpaceModel)

Query the results of the optimization called by `fit!`.
"""
results(model::StateSpaceModel) = model.results
function Base.isempty(results::Results)
    return isempty(results.coef_table) &&
           isnan(results.llk) &&
           isnan(results.aic) &&
           isnan(results.aicc) &&
           isnan(results.bic) &&
           iszero(results.num_observations) &&
           iszero(results.num_hyperparameters)
end

function fill_results!(model::StateSpaceModel, llk::Fl, std_err::Vector{Fl}) where Fl
    n_obs = length(model.system.y)
    num_hyperparameters = number_free_hyperparameters(model)
    coef_table = build_coef_table(model, std_err)
    # Fill results
    model.results.model_name = model_name(model)
    model.results.coef_table = coef_table
    model.results.llk = llk
    model.results.aic = aic(num_hyperparameters, llk)
    model.results.aicc = aicc(n_obs, num_hyperparameters, llk)
    model.results.bic = bic(n_obs, num_hyperparameters, llk)
    model.results.num_observations = n_obs
    model.results.num_hyperparameters = num_hyperparameters
    return model
end

function aic(k::Int, llk::Fl) where Fl
    return convert(Fl, 2 * k - 2 * llk)
end

function aicc(n::Int, k::Int, llk::Fl) where Fl
    return convert(Fl, aic(k, llk) + (2 * k * (k + 1) / (n - k - 1)))
end

function bic(n::Int, k::Int, llk::Fl) where Fl
    return convert(Fl, log(n) * k - 2 * llk)
end

aic(model::StateSpaceModel) = model.results.aic
aicc(model::StateSpaceModel) = model.results.aicc
bic(model::StateSpaceModel) = model.results.bic

function build_coef_table(model::StateSpaceModel, std_err::Vector{Fl}) where Fl
    all_coef = get_constrained_values(model)
    all_std_err = handle_std_err(model, std_err)
    all_z_stat = handle_z_stat(all_coef, all_std_err)
    all_p_value = handle_p_value(all_coef, all_std_err, all_z_stat)

    return CoefficientTable{Fl}(
        get_names(model), all_coef, all_std_err, all_z_stat, all_p_value
    )
end

function handle_std_err(model::StateSpaceModel, std_err::Vector{Fl}) where Fl
    all_std_err = fill(NaN, number_hyperparameters(model))
    # Put the std_err in the correct position
    for i in 1:length(std_err)
        hyperparamter_on_minimizer = model.hyperparameters.minimizer_hyperparameter_position[i]
        position_on_list_of_hyperparameters = position(
            hyperparamter_on_minimizer, model.hyperparameters
        )
        all_std_err[position_on_list_of_hyperparameters] = std_err[i]
    end
    return Fl.(all_std_err)
end

function handle_z_stat(all_coef::Vector{Fl}, all_std_err::Vector{Fl}) where Fl
    all_z_stat = fill(NaN, length(all_std_err))
    for i in 1:length(all_std_err)
        if !isnan(all_std_err[i])
            all_z_stat[i] = all_coef[i] / all_std_err[i]
        end
    end
    return Fl.(all_z_stat)
end

function handle_p_value(
    all_coef::Vector{Fl}, all_std_err::Vector{Fl}, all_z_stat::Vector{Fl}
) where Fl
    all_p_value = fill(NaN, length(all_std_err))
    for i in 1:length(all_std_err)
        if !isnan(all_std_err[i]) && all_std_err[i] > 0
            dist = Normal(all_coef[i], all_std_err[i])
            all_p_value[i] = 1 - 2 * abs(cdf(dist, all_z_stat[i]) - 0.5)
        end
    end
    return Fl.(all_p_value)
end
