export fit

function fit(model::StateSpaceModel;
             filter::KalmanFilter = default_filter(model),
             optimizer::Optimizer = Optimizer(Optim.LBFGS()))
    return fit(model, filter, optimizer)
end

function fit(model::StateSpaceModel, 
             filter::KalmanFilter,
             optimizer::Optimizer)
    
    initial_unconstrained_hyperparameter = handle_optim_initial_hyperparameters(model)
    # Optimization
    # try
        func = TwiceDifferentiable(x -> -optim_loglike(model, filter, x), 
                                        initial_unconstrained_hyperparameter)
        opt = optimize(func, initial_unconstrained_hyperparameter, 
                       optimizer.method, optimizer.options)
        opt_loglikelihood = -opt.minimum
        opt_hyperparameters = opt.minimizer
    # catch err
    #     println(err)
    #     println("seed diverged")
    # end

    log_lik = opt_loglikelihood
    update_model_hyperparameters!(model, opt_hyperparameters)
    return log_lik
end