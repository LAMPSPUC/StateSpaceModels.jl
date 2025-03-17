typeof_model_elements(model::StateSpaceModel) = eltype(model.system.y)
num_states(model::StateSpaceModel) = num_states(system(model))
num_series(model::StateSpaceModel) = num_series(system(model))
system(model::StateSpaceModel) = model.system
isunivariate(model::StateSpaceModel) = isa(model.system.y, Vector)
model_name(model::StateSpaceModel) = "$(typeof(model))"
num_observations(model::StateSpaceModel) = length(model.system.y)
observations(model::StateSpaceModel) = model.system.y

function lagmat(y::Vector{Fl}, k::Int) where Fl
    X = Matrix{Fl}(undef, length(y) - k, k)
    for i in 1:k
        X[:, i] = ShiftedArrays.lag(y, i)[k + 1:end]
    end
    return X
end

function variance_of_valid_observations(y::Vector{Fl}) where Fl
    return var(filter(!isnan, y))
end

function mean_of_valid_observations(y::Vector{Fl}) where Fl
    return mean(filter(!isnan, y))
end

function assert_zero_missing_values(y::Vector)
    for el in y
        if isnan(el)
            return error("This model does not support missing values.")
        end
    end
    return nothing
end