export is_valid_statespacemodel

typeof_model_elements(model::StateSpaceModel) = eltype(model.system.y)

"""
TODO
"""
num_states(model::StateSpaceModel) = num_states(system(model))

"""
TODO
"""
system(model::StateSpaceModel) = model.system

"""
TODO

Verifica se o model tem todas as funções implementadas
"""
function is_valid_statespacemodel(model_type::Type{<:StateSpaceModel})
    # TODO error message to indicate which method is missing
    tuple_with_model_type = Tuple{model_type}
    hasmethod(default_filter, tuple_with_model_type)
    hasmethod(initial_hyperparameters!, tuple_with_model_type)
    hasmethod(constrain_hyperparameters!, tuple_with_model_type)
    hasmethod(unconstrain_hyperparameters!, tuple_with_model_type)
    hasmethod(update!, tuple_with_model_type)
    return true
end

# standard print 
function Base.show(io::IO, model::StateSpaceModel)
    print(io, "A $(typeof(model)) model")
    return
end