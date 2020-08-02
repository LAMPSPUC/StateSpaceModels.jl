export has_fit_methods

typeof_model_elements(model::StateSpaceModel) = eltype(model.system.y)

num_states(model::StateSpaceModel) = num_states(system(model))
system(model::StateSpaceModel) = model.system

"""
    has_fit_methods(model_type::Type{<:StateSpaceModel})

Verify if a certain `StateSpaceModel` has all the methods perform the fit method.
"""
function has_fit_methods(model_type::Type{<:StateSpaceModel})
    tuple_with_model_type = Tuple{model_type}
    hasmethod(default_filter, tuple_with_model_type)
    hasmethod(initial_hyperparameters!, tuple_with_model_type)
    hasmethod(constrain_hyperparameters!, tuple_with_model_type)
    hasmethod(unconstrain_hyperparameters!, tuple_with_model_type)
    hasmethod(update!, tuple_with_model_type)
    return true
end

function Base.show(io::IO, model::StateSpaceModel)
    print(io, typeof(model), " model")
    return
end