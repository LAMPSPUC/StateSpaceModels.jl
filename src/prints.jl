function Base.show(io::IO, results::Results{Fl}) where Fl
    if !isempty(results)
        println(io, "                         Results                        ")
        println(io, "========================================================")
        println(io, "Number of observations:       ", results.num_observations)
        println(io, "Number of unknown parameters: ", results.num_hyperparameters)
        println(io, "Log-likelihood:               ", @sprintf("%.2f", results.llk))
        println(io, "AIC:                          ", @sprintf("%.2f", results.aic))
        println(io, "BIC:                          ", @sprintf("%.2f", results.bic))
    else
        # Up to discussion of what should be the behaviour
        error("You must call fit! to get results.")
    end
    return nothing
end

function Base.show(io::IO, model::StateSpaceModel)
    print(io, typeof(model), " model")
    return
end