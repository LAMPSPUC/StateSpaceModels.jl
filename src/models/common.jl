"""
Unobserved component models are state space models 
that you can name components, each component has a proper meaning
"""
abstract type UnobservedComponentModel <: StateSpaceModel end

typeof_model_elements(model::StateSpaceModel) = eltype(model.system.y)

"""
"""
get_num_states(model::StateSpaceModel) = get_num_states(get_system(model))

"""
"""
get_system(model::StateSpaceModel) = model.system

"""
TODO

Verifica se o model tem todas as funções implementadas
"""
function validate_statespacemodel(model::StateSpaceModel)
    # TODO colocar mensagem de erro caso não tenha os métodos necessários
    tuple_with_model_type = Tuple{typeof(model)}
    hasmethod(default_filter, tuple_with_model_type)
    hasmethod(initial_hyperparameters!, tuple_with_model_type)
    hasmethod(constraint_hyperparameters!, tuple_with_model_type)
    hasmethod(unconstraint_hyperparameters!, tuple_with_model_type)
    hasmethod(update!, tuple_with_model_type)
    return 
end

# standard print 
function Base.show(io::IO, model::StateSpaceModel)
    print(io, "A $(typeof(model)) model")
    return
end