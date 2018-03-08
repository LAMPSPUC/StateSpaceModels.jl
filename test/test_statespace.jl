
# Tests
@testset "Strutural model tests" begin
    @testset "Constant signal test" begin
        y = ones(100, 1)
        ss = statespace(y, 2)

        @test ss.state.trend[5:end, 1] + ss.state.seasonal[5:end, 1] ≈ y[5:end] atol = 1e-5
        @test sum(ss.param.sqrtH .< 1e-8) == length(ss.param.sqrtH)
        @test sum(ss.param.sqrtQ .< 1e-8) == length(ss.param.sqrtQ)
        println("sqrtH = $(ss.param.sqrtH)")
        println("sqrtQ = $(ss.param.sqrtQ)")
    end
    @testset "Linear signal test" begin
        y = Array{Float64}(0.1:0.1:5)
        ss = statespace(y, 2)

        @test ss.state.trend[4:end, 1] + ss.state.seasonal[4:end, 1] ≈ y[4:end] atol = 1e-5
        @test sum(ss.param.sqrtH .< 1e-8) == length(ss.param.sqrtH)
        @test sum(ss.param.sqrtQ .< 1e-8) == length(ss.param.sqrtQ)
    end
    @testset "Triangular signal test" begin
        y = Array{Float64}([collect(1:5); collect(4:-1:1); collect(2:5); collect(4:-1:1); collect(2:5);
                            collect(4:-1:1); collect(2:5); collect(4:-1:1); collect(2:5);
                            collect(4:-1:1); collect(2:5); collect(4:-1:1); collect(2:5);
                            collect(4:-1:1); collect(2:5); collect(4:-1:1); collect(2:5)])
        ss = statespace(y, 8)
        n = length(y)
        correct_trend = 3*ones(n)
        correct_slope = zeros(n)
        correct_seasonal = y - correct_trend

        @test ss.state.trend[11:end, 1] ≈ correct_trend[11:end] atol = 1e-5
        @test ss.state.slope[11:end, 1] ≈ correct_slope[11:end] atol = 1e-5
        @test ss.state.seasonal[11:end, 1] ≈ correct_seasonal[11:end] atol = 1e-5
    end
end
