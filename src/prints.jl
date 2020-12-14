function Base.show(io::IO, results::Results{Fl}) where Fl
    if !isempty(results)
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

function Base.show(io::IO, model::StateSpaceModel)
    print(io, typeof(model), " model")
    return nothing
end

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
    p_c = @sprintf("%.4f", coef_table.coef[offset])
    p_std = if isnan(coef_table.std_err[offset])
        " - "
    else
        @sprintf("%.3f", coef_table.std_err[offset])
    end
    p_z_stat = isnan(coef_table.z[offset]) ? " - " : @sprintf("%.3f", coef_table.z[offset])
    p_p_val = if isnan(coef_table.p_value[offset])
        " - "
    else
        @sprintf("%.3f", coef_table.p_value[offset])
    end
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
