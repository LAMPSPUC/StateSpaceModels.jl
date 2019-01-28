# Tests
@testset "Strutural model tests" begin
    @testset "Constant signal test" begin
        y = ones(15)
        ss = statespace(y, 2; nseeds = 1)

        correct_trend = y
        correct_slope = zeros(15)
        correct_seasonal = zeros(15)

        @test ss.state.trend[5:end, 1] ≈ correct_trend[5:end] atol = 1e-4
        @test ss.state.slope[5:end, 1] ≈ correct_slope[5:end] atol = 1e-4
        @test ss.state.seasonal[5:end, 1] ≈ correct_seasonal[5:end] atol = 1e-4
        @test sum(ss.param.sqrtH .< 1e-6) == length(ss.param.sqrtH)
        @test sum(ss.param.sqrtQ .< 1e-6) == length(ss.param.sqrtQ)
    end
    @testset "Linear signal test with simulation" begin
        y = Array{Float64}(0.1:0.1:1.5)
        ss = statespace(y, 2; nseeds = 1)

        correct_trend = y
        correct_slope = 0.1*ones(15)
        correct_seasonal = zeros(15)

        @test ss.state.trend[5:end, 1] ≈ correct_trend[5:end] atol = 1e-4
        @test ss.state.slope[5:end, 1] ≈ correct_slope[5:end] atol = 1e-4
        @test ss.state.seasonal[5:end, 1] ≈ correct_seasonal[5:end] atol = 1e-4
        @test sum(ss.param.sqrtH .< 1e-6) == length(ss.param.sqrtH)
        @test sum(ss.param.sqrtQ .< 1e-6) == length(ss.param.sqrtQ)

        N = 10
        sim = simulate(ss, N, 1000)
        q5 = zeros(N)
        q95 = zeros(N)
        for t = 1 : N
            q5[t] = quantile(sim[1, t,:], 0.05)
            q95[t] = quantile(sim[1, t,:], 0.95)
        end
        @test sum(q95 .- q5 .>= 0) == 10
        @test maximum(q95 .- q5) <= 1e-4
    end
    @testset "Basic multivariate test" begin
        y = [ones(20) collect(1:20)]
        ss = statespace(y, 2; nseeds = 5)
        correct_trend = y
        correct_slope = [zeros(20) ones(20)]
        correct_seasonal = [zeros(20) zeros(20)]

        N = 50
        S = 1000
        sim = simulate(ss, N, S)

        mean_sim1 = mean(sim[1, :, :], dims = 2)
        mean_sim2 = mean(sim[2, :, :], dims = 2)

        @test ss.state.trend[5:end, :] ≈ correct_trend[5:end, :] atol = 1e-4
        @test ss.state.slope[5:end, :] ≈ correct_slope[5:end, :] atol = 1e-4
        @test ss.state.seasonal[5:end, :] ≈ correct_seasonal[5:end, :] atol = 1e-4
        @test mean_sim1 ≈ ones(N) atol = 1e-2
        @test mean_sim2 ≈ collect(21.0:20.0 + N) atol = 1e-2
    end
end
