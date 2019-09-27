function print_header(verbose::Int)
    if verbose > 0
        println("==============================================================")
        println("                  StateSpaceModels.jl v0.2.3                  ")
        println(" (c) Raphael Saavedra, Guilherme Bodin, and Mario Souto, 2019 ")
        println("--------------------------------------------------------------")
        println("            Starting state-space model estimation.            ")
        
    end
end

function print_estimation_start(verbose::Int, nseeds::Int)
    if verbose > 0
        println("    Initiating maximum likelihood estimation with $(nseeds-1) seeds.    ") 
        println("--------------------------------------------------------------")
        println("             Seed 0 is aimed at degenerate cases.             ")
        println("--------------------------------------------------------------")
        println("||    seed    |     log-likelihood      |      time (s)     ||")
    end
end

function print_loglikelihood(verbose::Int, iseed::Int, loglikelihood::Vector{Float64}, t0::DateTime)
    if verbose > 0
        t1 = now() - t0

        str_seed   = @sprintf("%d", iseed-1) * "    |"
        str_loglik = @sprintf("%.4f", loglikelihood[iseed]) * "      |"
        str_time   = @sprintf("%.2f", t1.value/1000) * "     ||"

        str = "||"
        str *= " "^max(0, 13 - length(str_seed)) * str_seed
        str *= " "^max(0, 26 - length(str_loglik)) * str_loglik
        str *= " "^max(0, 21 - length(str_time)) * str_time

        println(str)
    end
end

function print_estimation_end(verbose::Int, loglikelihood::Vector{Float64})
    if verbose > 0
        str = "                  Log-likelihood: "
        str_loglik = @sprintf("%.4f", maximum(loglikelihood)) * "                  "
        str *= " "^max(0, 62 - length(str_loglik) - length(str)) * str_loglik

        println("--------------------------------------------------------------")
        println("           Maximum likelihood estimation complete.            ")
        println(str)
    end
end

function print_bottom(verbose::Int)
    if verbose > 0
        println("             End of state-space model estimation.             ")
        println("==============================================================")
    end
end
