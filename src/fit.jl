"""
    fit!(
        model::StateSpaceModel;
        filter::KalmanFilter=default_filter(model),
        optimizer::Optimizer=Optimizer(Optim.LBFGS())
    )

Estimate the state-space model parameters via maximum likelihood. The resulting optimal
hyperparameters and the corresponding log-likelihood are stored within the model.
"""
function fit!(
    model::StateSpaceModel;
    filter::KalmanFilter=default_filter(model),
    optimizer::Optimizer=Optimizer(Optim.LBFGS())
)

    initial_unconstrained_hyperparameter = handle_optim_initial_hyperparameters(model)
    # Should there be a try catch?
    func = TwiceDifferentiable(x -> -optim_loglike(model, filter, x),
                                    initial_unconstrained_hyperparameter)
    opt = optimize(func, initial_unconstrained_hyperparameter,
                    optimizer.method, optimizer.options)
    opt_loglikelihood   = -opt.minimum
    opt_hyperparameters = opt.minimizer
    update_model_hyperparameters!(model, opt_hyperparameters)
    numerical_hessian = Optim.hessian!(func, opt_hyperparameters)
    std_err = diag(inv(numerical_hessian))
    fill_results!(model, opt_loglikelihood, std_err)
    return nothing
end

function fill_results!(model::StateSpaceModel,
                       llk::Fl,
                       std_err::Vector{Fl}) where Fl
    n_obs               = length(model.system.y)
    num_hyperparameters = number_free_hyperparameters(model)
    coef_table          = build_coef_table(model, std_err)
    # Fill results
    model.results.coef_table          = coef_table
    model.results.llk                 = llk
    model.results.aic                 = AIC(num_hyperparameters, llk)
    model.results.bic                 = BIC(n_obs, num_hyperparameters, llk)
    model.results.num_observations    = n_obs
    model.results.num_hyperparameters = num_hyperparameters
    return
end

AIC(n_free_hyperparameters::Int, llk::Fl) where Fl = Fl(2 * n_free_hyperparameters - 2 * llk)
BIC(n::Int, n_free_hyperparameters::Int, llk::Fl) where Fl = Fl(log(n) * n_free_hyperparameters - 2 * llk)

function build_coef_table(model::StateSpaceModel,
                          std_err::Vector{Fl}) where Fl

    all_coef    = get_constrained_values(model)
    all_std_err = handle_std_err(model, std_err)
    all_z_stat  = handle_z_stat(all_coef, all_std_err)
    all_p_value = handle_p_value(all_coef, all_std_err, all_z_stat)

    return CoefficientTable{Fl}(
        get_names(model),
        all_coef,
        all_std_err,
        all_z_stat,
        all_p_value
    )
end

function handle_std_err(model::StateSpaceModel,
                        std_err::Vector{Fl}) where Fl
    all_std_err = fill(NaN, number_hyperparameters(model))
    # Put the std_err in the correct position
    for i in 1:length(std_err)
        hyperparamter_on_minimizer = model.hyperparameters.minimizer_hyperparameter_position[i]
        position_on_list_of_hyperparameters = position(hyperparamter_on_minimizer, model.hyperparameters)
        all_std_err[position_on_list_of_hyperparameters] = std_err[i]
    end
    return Fl.(all_std_err)
end

function handle_z_stat(all_coef::Vector{Fl},
                       all_std_err::Vector{Fl}) where Fl
    all_z_stat = fill(NaN, length(all_std_err))
    for i in 1:length(all_std_err)
        if !isnan(all_std_err[i])
            all_z_stat[i] = all_coef[i] / all_std_err[i]
        end
    end
    return Fl.(all_z_stat)
end

function handle_p_value(all_coef::Vector{Fl},
                        all_std_err::Vector{Fl},
                        all_z_stat::Vector{Fl}) where Fl
    all_p_value = fill(NaN, length(all_std_err))
    for i in 1:length(all_std_err)
        if !isnan(all_std_err[i]) && all_std_err[i] > 0
            dist = Normal(all_coef[i], all_std_err[i])
            all_p_value[i] = 1 - 2*abs(cdf(dist, all_z_stat[i]) - 0.5)
        end
    end
    return Fl.(all_p_value)
end
