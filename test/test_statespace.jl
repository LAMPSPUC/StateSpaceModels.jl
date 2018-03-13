# Tests
@testset "Strutural model tests" begin
    @testset "Constant signal test" begin
        y = ones(100, 1)
        ss = statespace(y, 2; nseeds = 5)

        @test ss.state.trend[5:end, 1] + ss.state.seasonal[5:end, 1] ≈ y[5:end] atol = 1e-5
        @test sum(ss.param.sqrtH .< 1e-7) == length(ss.param.sqrtH)
        @test sum(ss.param.sqrtQ .< 1e-7) == length(ss.param.sqrtQ)
    end
    @testset "Linear signal test" begin
        y = Array{Float64}(0.1:0.1:5)
        ss = statespace(y, 2; nseeds = 5)

        @test ss.state.trend[4:end, 1] + ss.state.seasonal[4:end, 1] ≈ y[4:end] atol = 1e-5
        @test sum(ss.param.sqrtH .< 1e-7) == length(ss.param.sqrtH)
        @test sum(ss.param.sqrtQ .< 1e-7) == length(ss.param.sqrtQ)
    end
    @testset "Triangular signal test" begin
        y = Array{Float64}([collect(1:5); collect(4:-1:1); collect(2:5); collect(4:-1:1); collect(2:5);
                            collect(4:-1:1); collect(2:5); collect(4:-1:1); collect(2:5);
                            collect(4:-1:1); collect(2:5); collect(4:-1:1); collect(2:5);
                            collect(4:-1:1); collect(2:5); collect(4:-1:1); collect(2:5)])
        ss = statespace(y, 8; nseeds = 5)
        n = length(y)
        correct_trend = 3*ones(n)
        correct_slope = zeros(n)
        correct_seasonal = y - correct_trend

        @test ss.state.trend[11:end, 1] ≈ correct_trend[11:end] atol = 1e-5
        @test ss.state.slope[11:end, 1] ≈ correct_slope[11:end] atol = 1e-5
        @test ss.state.seasonal[11:end, 1] ≈ correct_seasonal[11:end] atol = 1e-5
    end
    @testset "Basic multivariate test" begin
        y = [ones(50) collect(1:50)]
        ss = statespace(y, 3; nseeds = 5)
        correct_trend = y
        correct_slope = [zeros(50) ones(50)]
        correct_seasonal = [zeros(50) zeros(50)]

        @test ss.state.trend[4:end, :] ≈ correct_trend[4:end, :] atol = 1e-5
        @test ss.state.slope[4:end, :] ≈ correct_slope[4:end, :] atol = 1e-5
        @test ss.state.seasonal[4:end, :] ≈ correct_seasonal[4:end, :] atol = 1e-5
    end
end
