"""
TODO
"""
mutable struct HyperParameters{Fl <: AbstractFloat}
    num::Int
    names::Vector{String}
    # Poistion of the hyperparameter on the constrained and uncontrained values
    position::Dict{String, Int}
    # Minimizer vector corresponds to which hyperparameter
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

function fill_minimizer_hyperparameter_position!(model::StateSpaceModel) 
    fill_minimizer_hyperparameter_position!(model.hyperparameters)
    return
end
function fill_minimizer_hyperparameter_position!(hyperparameters::HyperParameters)
    # Reset minimizer hyperparameter_position
    hyperparameters.minimizer_hyperparameter_position = Dict{Int, String}()
    i = 1
    for name in get_names(hyperparameters)
        if is_free(name, hyperparameters)
            hyperparameters.minimizer_hyperparameter_position[i] = name
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
    # Associate the position of each index of the optimizer vector with a hyperparameter
    fill_minimizer_hyperparameter_position!(model)
    # Register the unconstrained initial_parameters
    unconstrain_hyperparameters!(model)
    # Return the unconstrained values of the initial_parameters
    return get_free_unconstrained_values(model)
end

get_hyperparameters(model::StateSpaceModel) = model.hyperparameters

"""
    get_names(model::StateSpaceModel)

Get the names of the hyperparameters registered on a `StateSpaceModel`.
"""
get_names(model::StateSpaceModel) = get_names(model.hyperparameters)

"""
    number_hyperparameters(model::StateSpaceModel)

Get the number of hyperparameters registered on a `StateSpaceModel`.
"""
number_hyperparameters(model::StateSpaceModel) = number_hyperparameters(model.hyperparameters)
number_hyperparameters(hyperparameters::HyperParameters) = hyperparameters.num

position(str::String, hyperparameters::HyperParameters) = hyperparameters.position[str]
get_position(hyperparameters::HyperParameters) = hyperparameters.position
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
    for (k, v) in hyperparameters.minimizer_hyperparameter_position
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
    set_initial_hyperparameters!(model::StateSpaceModel,
                                 initial_hyperparameters::Dict{String, <:Real})

Fill a model with user inputed initial points for hyperparameter optimzation.

# Example
```jldoctest
julia> model = LocalLevel(rand(100));
```
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
    fix_hyperparameters!(model::StateSpaceModel, fixed_hyperparameters::Dict)

Fixes the desired hyperparameters so that they are not considered as decision variables in
the model estimation.

# Example
```jldoctest
julia> model = LocalLevel(rand(100));

julia> get_names(model)
2-element Array{String,1}:
 "sigma2_ε"
 "sigma2_η"

julia> fix_hyperparameters!(model, Dict("sigma2_ε" => 15099.0))

```
"""
function fix_hyperparameters! end

function fix_hyperparameters!(model::StateSpaceModel, fixed_hyperparameters::Dict)
    return fix_hyperparameters!(model.hyperparameters, fixed_hyperparameters)
end
function fix_hyperparameters!(hyperparameters::HyperParameters{Fl},
                              fixed_hyperparameters::Dict{String, Fl}) where Fl
    # assert we have these hyperparameters
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

"""
    isfitted(model::StateSpaceModel) -> Bool

Verify if `model` is fitted, i.e., returns `false` if there is at least one `NaN` entry in
the hyperparameters.
"""
function isfitted(model::StateSpaceModel)
    c1 = any(isnan, model.hyperparameters.unconstrained_values)
    c2 = any(isnan, model.hyperparameters.constrained_values)
    if c1 || c2
        return false
    else
        return true
    end
end

# Some special constraint functions
"""
    constrain_box!(model::StateSpaceModel, str::String, lb::Fl, ub::Fl) where Fl

Map a constrained hyperparameter ``\\psi \\in [lb, ub]`` to an unconstrained hyperparameter 
``\\psi_* \\in \\mathbb{R}``.

The mapping is
``\\psi = lb + \\frac{ub - lb}{1 + \\exp{-\\psi_*}}``
"""
function constrain_box!(model::StateSpaceModel, str::String, lb::Fl, ub::Fl) where Fl
    update_constrained_value!(model, str, lb + ((ub - lb)/(1 + exp(-get_unconstrained_value(model, str)))))
    return
end

"""
    unconstrain_box!(model::StateSpaceModel, str::String, lb::Fl, ub::Fl) where Fl

Map an unconstrained hyperparameter ``\\psi_* \\in \\mathbb{R}`` to a constrained hyperparameter 
``\\psi \\in [lb, ub]``.

The mapping is
``\\psi_* = -\\ln \\frac{ub - lb}{\\psi - lb} - 1``
"""
function unconstrain_box!(model::StateSpaceModel, str::String, lb::Fl, ub::Fl) where Fl
    update_unconstrained_value!(model, str, -log((ub - lb)/(get_constrained_value(model, str) - lb) - 1))
    return
end

"""
    constrain_variance!(model::StateSpaceModel, str::String)

Map a constrained hyperparameter ``\\psi \\in \\mathbb{R}^+`` to an unconstrained hyperparameter 
``\\psi_* \\in \\mathbb{R}``.

The mapping is
``\\psi = \\psi_*^2``
"""
function constrain_variance!(model::StateSpaceModel, str::String)
    update_constrained_value!(model, str, (get_unconstrained_value(model, str))^2)
    return
end

"""
    unconstrain_variance!(model::StateSpaceModel, str::String)

Map an unconstrained hyperparameter ``\\psi_* \\in \\mathbb{R}`` to a constrained hyperparameter 
``\\psi \\in \\mathbb{R}^+``.

The mapping is
``\\psi_* = \\sqrt{\\psi_}``
"""
function unconstrain_variance!(model::StateSpaceModel, str::String)
    update_unconstrained_value!(model, str, sqrt(get_constrained_value(model, str)))
    return
end

"""
    constrain_identity!(model::StateSpaceModel, str::String)

Map an constrained hyperparameter ``\\psi \\in \\mathbb{R}`` to an unconstrained hyperparameter 
``\\psi_* \\in \\mathbb{R}``. This function is necessary to copy values from a location to another
inside `HyperParameters`

The mapping is
``\\psi = \\psi_*``
"""
function constrain_identity!(model::StateSpaceModel, str::String)
    update_constrained_value!(model, str, get_unconstrained_value(model, str))
    return
end

"""
    unconstrain_identity!(model::StateSpaceModel, str::String)

Map an unconstrained hyperparameter ``\\psi_* \\in \\mathbb{R}`` to a constrained hyperparameter 
``\\psi \\in \\mathbb{R}``. This function is necessary to copy values from a location to another
inside `HyperParameters`

The mapping is
``\\psi_* = \\psi``
"""
function unconstrain_identity!(model::StateSpaceModel, str::String)
    update_unconstrained_value!(model, str, get_constrained_value(model, str))
    return
end
