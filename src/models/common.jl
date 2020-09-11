export has_fit_methods, results

typeof_model_elements(model::StateSpaceModel) = eltype(model.system.y)
num_states(model::StateSpaceModel) = num_states(system(model))
system(model::StateSpaceModel) = model.system

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

mutable struct Results{Fl <: AbstractFloat}
    coef_table::CoefficientTable{Fl}
    llk::Fl
    aic::Fl
    bic::Fl
    num_observations::Int
    num_hyperparameters::Int
    fitted::Bool
end
function Results{Fl}() where Fl
    return Results{Fl}(
        CoefficientTable{Fl}(),
        Fl(NaN),
        Fl(NaN),
        Fl(NaN),
        0,
        0,
        false
    )
end
results(model::StateSpaceModel) = model.results

function print_coef_table(io::IO, coef_table::CoefficientTable{Fl}) where Fl
    println(io, "--------------------------------------------------------")
    println(io, "Parameter      Estimate   Std.Error     z stat   p-value")
    offset = 1
    for i in 1:length(coef_table)
        p_c, p_std, p_z_stat, p_p_val = print_coef(coef_table, offset)
        println(io, build_print(p_c, p_std, p_z_stat, p_p_val, coef_table.names[offset]))
        offset += 1
    end
    return nothing
end

function print_coef(coef_table::CoefficientTable{Fl}, offset::Int) where Fl
    p_c      = @sprintf("%.4f", coef_table.coef[offset])
    p_std    = @sprintf("%.4f", coef_table.std_err[offset])
    p_z_stat = @sprintf("%.4f", coef_table.z[offset])
    p_p_val  = @sprintf("%.4f", coef_table.p_value[offset])
    return p_c, p_std, p_z_stat, p_p_val
end

function build_print(p_c, p_std, p_z_stat, p_p_val, param)
    p = ""
    p *= param
    p *= " "^max(0, 23 - length(p_c) - length(param))
    p *= p_c
    p *= " "^max(0, 12 - length(p_std))
    p *= p_std
    p *= " "^max(0, 11 - length(p_z_stat))
    p *= p_z_stat
    p *= " "^max(0, 10 - length(p_p_val))
    p *= p_p_val
    return p
end

function Base.show(io::IO, results::Results{Fl}) where Fl
    if results.fitted
        println(io, "                         Results                        ")
        println(io, "========================================================")
        println(io, "Number of observations:       ", results.num_observations)
        println(io, "Number of unknown parameters: ", results.num_hyperparameters)
        println(io, "Log-likelihood:               ", @sprintf("%.2f", results.llk))
        println(io, "AIC:                          ", @sprintf("%.2f", results.aic))
        println(io, "BIC:                          ", @sprintf("%.2f", results.bic))
        print_coef_table(io, results.coef_table)
    else
        # Up to discussion of what should be the behaviour
        error("You must call fit! to get results.")
    end
    return nothing
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

function Base.show(io::IO, model::StateSpaceModel)
    print(io, typeof(model), " model")
    return
end
