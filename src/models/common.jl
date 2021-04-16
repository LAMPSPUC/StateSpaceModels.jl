typeof_model_elements(model::StateSpaceModel) = eltype(model.system.y)
num_states(model::StateSpaceModel) = num_states(system(model))
system(model::StateSpaceModel) = model.system
isunivariate(model::StateSpaceModel) = isa(model.system.y, Vector)
model_name(model::StateSpaceModel) = "$(typeof(model))"
length_observations(model::StateSpaceModel) = length(model.system.y)
observations(model::StateSpaceModel) = model.system.y