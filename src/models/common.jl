typeof_model_elements(model::StateSpaceModel) = eltype(model.system.y)
num_states(model::StateSpaceModel) = num_states(system(model))
system(model::StateSpaceModel) = model.system
isunivariate(model::StateSpaceModel) = isa(model.system.y, Vector)
isunivariate(y::Array) = isa(y, Vector)

struct CoefficientTable{Fl <: AbstractFloat}
    names::Vector{String}
    coef::Vector{Fl}
    std_err::Vector{Fl}
    z::Vector{Fl}
    p_value::Vector{Fl}

    function CoefficientTable{Fl}(names::Vector{String},
                                coef::Vector{Fl},
                                std_err::Vector{Fl},
                                z::Vector{Fl},
                                p_value::Vector{Fl}) where Fl
        @assert length(names) == length(coef) == length(std_err) == length(z) == length(p_value)
        return new{Fl}(names, coef, std_err, z, p_value)
    end
end
function CoefficientTable{Fl}() where Fl
    return CoefficientTable{Fl}(
                        String[],
                        Fl[],
                        Fl[],
                        Fl[],
                        Fl[])
end
Base.length(coef_table::CoefficientTable) = length(coef_table.names)
function Base.isempty(coef_table::CoefficientTable)
    return isempty(coef_table.names) &&
        isempty(coef_table.coef) &&
        isempty(coef_table.std_err) &&
        isempty(coef_table.z) &&
        isempty(coef_table.p_value)
end

mutable struct Results{Fl <: AbstractFloat}
    coef_table::CoefficientTable{Fl}
    llk::Fl
    aic::Fl
    bic::Fl
    num_observations::Int
    num_hyperparameters::Int
end
function Results{Fl}() where Fl
    return Results{Fl}(
        CoefficientTable{Fl}(),
        Fl(NaN),
        Fl(NaN),
        Fl(NaN),
        0,
        0
    )
end

"""
    results(model::StateSpaceModel)

Query the results of the optimizaation called by `fit!`.
"""
results(model::StateSpaceModel) = model.results
function Base.isempty(results::Results)
    return isempty(results.coef_table) &&
           isnan(results.llk) &&
           isnan(results.aic) &&
           isnan(results.bic) &&
           iszero(results.num_observations) &&
           iszero(results.num_hyperparameters)
end

"""
    has_fit_methods(model_type::Type{<:StateSpaceModel}) -> Bool

Verify if a certain `StateSpaceModel` has the necessary methods to perform `fit!``.
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
