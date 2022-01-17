"""
    print_results(model::StateSpaceModel)

Query the results of the optimization called by `fit!`.
"""
function print_results(model::StateSpaceModel)
    results = get_results(model)
    if !isempty(results)
        println("                             Results                           ")
        println("===============================================================" * adjust_charachters("=", results.coef_table))
        println("Model:                        ", results.model_name)
        println("Number of observations:       ", results.num_observations)
        println("Number of unknown parameters: ", results.num_hyperparameters)
        println("Log-likelihood:               ", pretty_number(results.llk))
        println("AIC:                          ", pretty_number(results.aic))
        println("AICc:                         ", pretty_number(results.aicc))
        println("BIC:                          ", pretty_number(results.bic))
        print_coef_table(results.coef_table)
    else
        println("Results are empty.")
    end
end

function Base.show(io::IO, model::StateSpaceModel)
    print(io, model_name(model))
    return nothing
end

function print_coef_table(coef_table::CoefficientTable{Fl}) where Fl
    # The repeats adjust the space for large names
    println("---------------------------------------------------------------" * adjust_charachters("-", coef_table))
    println("Parameter" * adjust_charachters(" ", coef_table) * "      Estimate      Std.Error      z stat      p-value")
    offset = 1
    for i in 1:length(coef_table)
        p_c, p_std, p_z_stat, p_p_val = print_coef(coef_table, offset)
        println(build_print(p_c, p_std, p_z_stat, p_p_val, offset, coef_table))
        offset += 1
    end
    return nothing
end

function print_coef(coef_table::CoefficientTable{Fl}, offset::Int) where Fl
    p_c = pretty_number(coef_table.coef[offset])
    p_std = pretty_number(coef_table.std_err[offset])
    p_z_stat = pretty_number(coef_table.z[offset])
    p_p_val = pretty_number(coef_table.p_value[offset])
    return p_c, p_std, p_z_stat, p_p_val
end

function build_print(p_c, p_std, p_z_stat, p_p_val, offset, coef_table)
    max_len_name = maximum(length.(coef_table.names))
    p = ""
    p *= coef_table.names[offset]
    p *= " "^max(0, 23 + max(0, max_len_name - 9) - length(p_c) - length(coef_table.names[offset]))
    p *= p_c
    p *= " "^max(0, 15 - length(p_std))
    p *= p_std
    p *= " "^max(0, 12 - length(p_z_stat))
    p *= p_z_stat
    p *= " "^max(0, 13 - length(p_p_val))
    p *= p_p_val
    return p
end

function pretty_number(num::Number)
    if isnan(num)
        return " - "
    elseif abs(num) > 1e4
        return @sprintf("%.3E", num)
    else
        return @sprintf("%.4f", num)
    end
end

function adjust_charachters(str::String, coef_table::CoefficientTable)
    max_len_name = maximum(length.(coef_table.names))
    return repeat(str, max(0, max_len_name - 9))
end