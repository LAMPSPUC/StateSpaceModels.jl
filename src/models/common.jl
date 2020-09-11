export has_fit_methods, results

typeof_model_elements(model::StateSpaceModel) = eltype(model.system.y)
num_states(model::StateSpaceModel) = num_states(system(model))
system(model::StateSpaceModel) = model.system

mutable struct Results{Fl <: AbstractFloat}
    llk::Fl
    aic::Fl
    bic::Fl
    num_observations::Int
    num_hyperparameters::Int
end
function Results{Fl}() where Fl
    return Results{Fl}(
        Fl(NaN),
        Fl(NaN),
        Fl(NaN),
        0,
        0
    )
end
results(model::StateSpaceModel) = model.results
function Base.isempty(results::Results)
    return isnan(results.llk) && 
           isnan(results.aic) && 
           isnan(results.bic) &&
           iszero(results.num_observations) && 
           iszero(results.num_hyperparameters)
end

"""
    has_fit_methods(model_type::Type{<:StateSpaceModel}) -> Bool

Verify if a certain `StateSpaceModel` has the necessary methods to perform the fit!.
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
