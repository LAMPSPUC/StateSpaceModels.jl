typeof_model_elements(model::StateSpaceModel) = eltype(model.system.y)
num_states(model::StateSpaceModel) = num_states(system(model))
num_series(model::StateSpaceModel) = num_series(system(model))
system(model::StateSpaceModel) = model.system
isunivariate(model::StateSpaceModel) = isa(model.system.y, Vector)
model_name(model::StateSpaceModel) = "$(typeof(model))"

function lagmat(y::Vector{Fl}, k::Int) where Fl
    X = Matrix{Fl}(undef, length(y) - k, k)
    for i in 1:k
        X[:, i] = lag(y, i)[k + 1:end]
    end
    return X
end