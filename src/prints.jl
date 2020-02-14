function print_header(verbose::Int)
    if verbose > 0
        println("==============================================================")
        println("                  StateSpaceModels.jl v0.3.2                  ")
        println(" (c) Raphael Saavedra, Guilherme Bodin, and Mario Souto, 2019 ")
        println("--------------------------------------------------------------")
        println("            Starting state-space model estimation.            ")

    end
end

function print_estimation_start(verbose::Int, nseeds::Int)
    if verbose > 0
        println("    Initiating maximum likelihood estimation with $(nseeds) seeds.    ")
        println("--------------------------------------------------------------")
        println("||    seed    |     log-likelihood      |      time (s)     ||")
    end
end

function print_loglikelihood(verbose::Int, iseed::Int, loglikelihood::Vector{T}, t0::DateTime) where T
    if verbose > 0
        t1 = now() - t0

        str_seed   = @sprintf("%d", iseed) * "    |"
        str_loglik = @sprintf("%.4f", loglikelihood[iseed]) * "      |"
        str_time   = @sprintf("%.2f", t1.value/1000) * "     ||"

        str = "||"
        str *= " "^max(0, 13 - length(str_seed)) * str_seed
        str *= " "^max(0, 26 - length(str_loglik)) * str_loglik
        str *= " "^max(0, 21 - length(str_time)) * str_time

        println(str)
    end
end

function print_estimation_end(verbose::Int, log_lik::T, aic::T, bic::T) where T
    if verbose > 0
        str_1 = "                  Log-likelihood: "
        str_log_lik = @sprintf("%.4f", log_lik) * "                  "
        str_1 *= " "^max(0, 62 - length(str_log_lik) - length(str_1)) * str_log_lik

        str_2 = "                             AIC: "
        str_aic = @sprintf("%.4f", aic) * "                  "
        str_2 *= " "^max(0, 62 - length(str_aic) - length(str_2)) * str_aic

        str_3 = "                             BIC: "
        str_bic = @sprintf("%.4f", bic) * "                  "
        str_3 *= " "^max(0, 62 - length(str_bic) - length(str_3)) * str_bic

        println("--------------------------------------------------------------")
        println("           Maximum likelihood estimation complete.            ")
        println(str_1)
        println(str_2)
        println(str_3)
    end
end

function print_bottom(verbose::Int)
    if verbose > 0
        println("             End of state-space model estimation.             ")
        println("==============================================================")
    end
end
