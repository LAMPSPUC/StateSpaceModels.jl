export fit!

"""
TODO
"""
function fit!(model::StateSpaceModel; 
             filter::KalmanFilter = default_filter(model),
             optimizer::Optimizer = Optimizer(Optim.LBFGS()))
    
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
    n_obs = length(model.system.y)
    num_hyperparameters = number_free_hyperparameters(model)
    coef_table = build_coef_table(model, std_err)
    # Fill results
    model.results.coef_table          = coef_table
    model.results.llk                 = llk
    model.results.aic                 = AIC(num_hyperparameters, llk)
    model.results.bic                 = BIC(n_obs, num_hyperparameters, llk)
    model.results.num_observations    = n_obs
    model.results.num_hyperparameters = num_hyperparameters
    model.results.fitted              = true
    return
end

AIC(n_free_hyperparameters::Int, llk::Fl) where Fl = Fl(2 * n_free_hyperparameters - 2 * llk)
BIC(n::Int, n_free_hyperparameters::Int, llk::Fl) where Fl = Fl(log(n) * n_free_hyperparameters - 2 * llk)

function build_coef_table(model::StateSpaceModel,
                          std_err::Vector{Fl}) where Fl
    return CoefficientTable{Fl}(
        get_names(model),
        get_constrained_values(model),
        std_err,
        rand(length(get_names(model))),
        rand(length(get_names(model)))
    )
end