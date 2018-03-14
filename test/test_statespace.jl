# Tests
@testset "Strutural model tests" begin
    @testset "Constant signal test" begin
        y = ones(15)
        ss = statespace(y, 2; nseeds = 1)

        correct_trend = y
        correct_slope = zeros(15)
        correct_seasonal = zeros(15)

        @test ss.state.trend[4:end, 1] ≈ correct_trend[4:end] atol = 1e-5
        @test ss.state.slope[4:end, 1] ≈ correct_slope[4:end] atol = 1e-5
        @test ss.state.seasonal[4:end, 1] ≈ correct_seasonal[4:end] atol = 1e-5
        @test sum(ss.param.sqrtH .< 1e-6) == length(ss.param.sqrtH)
        @test sum(ss.param.sqrtQ .< 1e-6) == length(ss.param.sqrtQ)
    end
    @testset "Linear signal test with simulation" begin
        y = Array{Float64}(0.1:0.1:1.5)
        ss = statespace(y, 2; nseeds = 1)

        correct_trend = y
        correct_slope = 0.1*ones(15)
        correct_seasonal = zeros(15)

        @test ss.state.trend[4:end, 1] ≈ correct_trend[4:end] atol = 1e-5
        @test ss.state.slope[4:end, 1] ≈ correct_slope[4:end] atol = 1e-5
        @test ss.state.seasonal[4:end, 1] ≈ correct_seasonal[4:end] atol = 1e-5
        @test sum(ss.param.sqrtH .< 1e-6) == length(ss.param.sqrtH)
        @test sum(ss.param.sqrtQ .< 1e-6) == length(ss.param.sqrtQ)

        N = 10
        sim = simulate(ss, N, 1000)
        q5 = zeros(N)
        q95 = zeros(N)
        for t = 1 : N
            q5[t] = quantile(sim[t,:], 0.05)
            q95[t] = quantile(sim[t,:], 0.95)
        end
        @test sum(q95 .- q5 .>= 0) == 10
        @test maximum(q95 .- q5) <= 1e-5
    end
    @testset "Basic multivariate test" begin
        y = [ones(20) collect(1:20)]
        ss = statespace(y, 3; nseeds = 5)
        correct_trend = y
        correct_slope = [zeros(20) ones(20)]
        correct_seasonal = [zeros(20) zeros(20)]

        @test ss.state.trend[4:end, :] ≈ correct_trend[4:end, :] atol = 1e-5
        @test ss.state.slope[4:end, :] ≈ correct_slope[4:end, :] atol = 1e-5
        @test ss.state.seasonal[4:end, :] ≈ correct_seasonal[4:end, :] atol = 1e-5
    end
end
