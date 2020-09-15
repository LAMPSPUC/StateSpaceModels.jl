export fit!

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
    fill_results!(model, opt_loglikelihood)
    return nothing
end

function fill_results!(model::StateSpaceModel,
                       llk::Fl) where Fl
    n_obs = length(model.system.y)
    num_hyperparameters = number_free_hyperparameters(model)
    # Fill results
    model.results.llk                 = llk
    model.results.aic                 = AIC(num_hyperparameters, llk)
    model.results.bic                 = BIC(n_obs, num_hyperparameters, llk)
    model.results.num_observations    = n_obs
    model.results.num_hyperparameters = num_hyperparameters
    return
end

AIC(n_free_hyperparameters::Int, llk::Fl) where Fl = Fl(2 * n_free_hyperparameters - 2 * llk)
BIC(n::Int, n_free_hyperparameters::Int, llk::Fl) where Fl = Fl(log(n) * n_free_hyperparameters - 2 * llk)
