export get_constrained_value,
    get_hyperparameters,
    get_minimizer_hyperparameter_position,
    set_initial_hyperparameters!,
    fix_hyperparameters!


mutable struct HyperParameters{Fl <: AbstractFloat}
    num::Int
    names::Vector{String}
    # Poistion of the hyperparameter on the
    # constrained and uncontrained values
    position::Dict{String, Int}
    # Minimizer vector corresponds to which
    # hyperparameter
    minimizer_hyperparameter_position::Dict{Int, String}
    constrained_values::Vector{Fl}
    unconstrained_values::Vector{Fl}
    fixed_constrained_values::Dict{String, Fl}

    function HyperParameters{Fl}(names::Vector{String}) where {Fl}
        num                               = length(names)
        positions                         = Dict(names .=> collect(1:num))
        minimizer_hyperparameter_position = Dict{Int, String}()
        constrained_values                = fill(Fl(NaN), num)
        unconstrained_values              = fill(Fl(NaN), num)
        fixed_constrained_values          = Dict{String, Fl}()
        return new{Fl}(num,
                       names,
                       positions,
                       minimizer_hyperparameter_position,
                       constrained_values,
                       unconstrained_values,
                       fixed_constrained_values)
    end
end

function register_unconstrained_values!(model::StateSpaceModel,
                                        unconstrained_values::Vector{Fl}) where Fl
    return register_unconstrained_values!(model.hyperparameters, unconstrained_values)
end
function register_unconstrained_values!(hyperparameters::HyperParameters,
                                        unconstrained_values::Vector{Fl}) where Fl
    for (i, val) in enumerate(unconstrained_values)
        update_unconstrained_value!(hyperparameters,
                                    hyperparameters.minimizer_hyperparameter_position[i],
                                    val)
    end
    return nothing
end

#TODO same function but receiving hyperparameter
function fill_minimizer_hyperparameter_position!(model::StateSpaceModel)
    # Reset minimizer hyperparameter_position
    model.hyperparameters.minimizer_hyperparameter_position = Dict{Int, String}()
    i = 1
    for name in get_names(model.hyperparameters)
        if is_free(name, model.hyperparameters)
            model.hyperparameters.minimizer_hyperparameter_position[i] = name
            i += 1
        end
    end
    return
end

function handle_optim_initial_hyperparameters(model::StateSpaceModel)
    # Register default initial hyperparameters
    initial_hyperparameters!(model)
    # Possibly fix some user specified initial hyperparameters
    fix_hyperparameters!(model)
    # Associate the position of each index of the
    # optimizer vector with a hyperparameter
    fill_minimizer_hyperparameter_position!(model)
    # Register the unconstrained initial_parameters
    unconstrain_hyperparameters!(model)
    # Return the unconstrained values of the initial_parameters
    return get_free_unconstrained_values(model)
end

get_hyperparameters(model::StateSpaceModel) = model.hyperparameters
get_names(model::StateSpaceModel) = get_names(model.hyperparameters)

number_hyperparameters(model::StateSpaceModel) = number_hyperparameters(model.hyperparameters)
number_hyperparameters(hyperparameters::HyperParameters) = hyperparameters.num

position(str::String, hyperparameters::HyperParameters) = hyperparameters.position[str]
get_position(hyperparameters::HyperParameters) = hyperparameters.position
get_minimizer_hyperparameter_position(hyperparameters::HyperParameters) = hyperparameters.minimizer_hyperparameter_position
get_names(hyperparameters::HyperParameters) = hyperparameters.names
get_constrained_values(model::StateSpaceModel) = get_constrained_values(model.hyperparameters)
get_constrained_values(hyperparameters::HyperParameters) = hyperparameters.constrained_values
get_unconstrained_values(model::StateSpaceModel) = get_unconstrained_values(model.hyperparameters)
get_unconstrained_values(hyperparameters::HyperParameters) = hyperparameters.unconstrained_values
get_fixed_constrained_values(hyperparameters::HyperParameters) = hyperparameters.fixed_constrained_values

get_constrained_value(model::StateSpaceModel, str::String) = get_constrained_value(model.hyperparameters, str)
get_constrained_value(hyperparameters::HyperParameters, str::String) = hyperparameters.constrained_values[position(str, hyperparameters)]
update_constrained_value!(model::StateSpaceModel, str::String, value) = update_constrained_value!(model.hyperparameters, str, value)
function update_constrained_value!(hyperparameters::HyperParameters, str::String, value)
    hyperparameters.constrained_values[position(str, hyperparameters)] = value
end

get_unconstrained_value(model::StateSpaceModel, str::String) = get_unconstrained_value(model.hyperparameters, str)
get_unconstrained_value(hyperparameters::HyperParameters, str::String) = hyperparameters.unconstrained_values[position(str, hyperparameters)]
update_unconstrained_value!(model::StateSpaceModel, str::String, value) = update_unconstrained_value!(model.hyperparameters, str, value)
function update_unconstrained_value!(hyperparameters::HyperParameters, str::String, value)
    hyperparameters.unconstrained_values[position(str, hyperparameters)] = value
end

get_free_unconstrained_values(model::StateSpaceModel) = get_free_unconstrained_values(model.hyperparameters)
function get_free_unconstrained_values(hyperparameters::HyperParameters{Fl}) where Fl
    free_unconstrained_values = Vector{Fl}(undef, number_free_hyperparameters(hyperparameters))
    for (k, v) in get_minimizer_hyperparameter_position(hyperparameters)
        free_unconstrained_values[k] = get_unconstrained_value(hyperparameters, v)
    end
    return free_unconstrained_values
end

function has_hyperparameter(str::String, hyperparameters::HyperParameters)
    return haskey(hyperparameters.position, str)
end
function has_hyperparameter(str::Vector{String}, hyperparameters::HyperParameters)
    for s in str
        if !has_hyperparameter(s, hyperparameters)
            return false
        end
    end
    return true
end


"""
"""
function set_initial_hyperparameters!(model::StateSpaceModel,
                                      initial_hyperparameters::Dict{String, <:Real})
    for (k, v) in initial_hyperparameters
        update_constrained_value!(model, k, v)
    end
    return
end

function is_fixed(str::String, hyperparameters::HyperParameters)
    return haskey(hyperparameters.fixed_constrained_values, str)
end
number_free_hyperparameters(model::StateSpaceModel) = number_free_hyperparameters(model.hyperparameters)
function number_free_hyperparameters(hyperparameters::HyperParameters)
    return number_hyperparameters(hyperparameters) - number_fixed_hyperparameters(hyperparameters)
end
function is_free(str::String, hyperparameters::HyperParameters)
    return !is_fixed(str, hyperparameters)
end
number_fixed_hyperparameters(model::StateSpaceModel) = number_fixed_hyperparameters(model.hyperparameters)
function number_fixed_hyperparameters(hyperparameters::HyperParameters)
    return length(hyperparameters.fixed_constrained_values)
end

"""
    fix_hyperparameters!

Fixes the desired hyperparameters so that they are not considered as decision variables in
the model estimation.
"""
function fix_hyperparameters! end

function fix_hyperparameters!(model::StateSpaceModel, fixed_hyperparameters::Dict)
    return fix_hyperparameters!(model.hyperparameters, fixed_hyperparameters)
end

function fix_hyperparameters!(hyperparameters::HyperParameters{Fl},
                              fixed_hyperparameters::Dict{String, Fl}) where Fl
    # assert we have these hyperparameters
    #TODO falar qual está faltando como erro
    @assert has_hyperparameter(collect(keys(fixed_hyperparameters)), hyperparameters)
    # build dict
    hyperparameters.fixed_constrained_values = fixed_hyperparameters
    return
end

fix_hyperparameters!(model::StateSpaceModel) = fix_hyperparameters!(model.hyperparameters)
function fix_hyperparameters!(hyperparameters::HyperParameters{Fl}) where Fl
    for (name, value) in hyperparameters.fixed_constrained_values
        update_constrained_value!(hyperparameters, name, value)
    end
    return
end


# Some special constraint functions
"""
TODO
"""
function constrain_box!(model::StateSpaceModel, str::String, lb::Fl, ub::Fl) where Fl
    update_constrained_value!(model, str, lb + ((ub - lb)/(1 + exp(-get_unconstrained_value(model, str)))))
    return
end

"""
TODO
"""
function unconstrain_box!(model::StateSpaceModel, str::String, lb::Fl, ub::Fl) where Fl
    update_unconstrained_value!(model, str, -log((ub - lb)/(get_constrained_value(model, str) - lb - 1)))
    return
end

"""
TODO
"""
function constrain_variance!(model::StateSpaceModel, str::String)
    update_constrained_value!(model, str, (get_unconstrained_value(model, str))^2)
    return
end

"""
TODO
"""
function unconstrain_variance!(model::StateSpaceModel, str::String)
    update_unconstrained_value!(model, str, sqrt(get_constrained_value(model, str)))
    return
end

"""
TODO
"""
function constrain_identity!(model::StateSpaceModel, str::String)
    update_constrained_value!(model, str, get_unconstrained_value(model, str))
    return
end

"""
TODO
"""
function unconstrain_identity!(model::StateSpaceModel, str::String)
    update_unconstrained_value!(model, str, get_constrained_value(model, str))
    return
end
