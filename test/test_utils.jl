@testset "utils.jl" begin
    @testset "size_function" begin
        y = ones(15)
        X = randn(15, 2)
        s = 2
        model = structural(y, s; X = X)
        n, p, m, r = size(model)
        @test n == 15
        @test p == 1
        @test m == s + size(X, 2) + 1
        @test r == 3

        y = ones(15)
        X = randn(15, 15)
        s = 2
        model = structural(y, s; X = X)
        n, p, m, r = size(model)
        @test n == 15
        @test p == 1
        @test m == s + size(X, 2) + 1
        @test r == 3
    end

    @testset "ztr" begin
        y = ones(15)
        model = local_level(y)
        Z, T, R = ztr(model)
        @test all(Z .== [1.0])
        @test T == [1.0][:, :]
        @test R == [1.0][:, :]
    end

    @testset "gram" begin
        A = [1.0][:, :]
        @test A == SSM.gram(A)
        B = [1 1.0;0 1]
        @test B*B' == SSM.gram(B)
    end

    @testset "gram_in_time" begin
        C = Array{Float64, 3}(undef, 2, 2, 5)
        B = [1 1.0;0 1]
        C[:, :, 1:5] .= B
        @test B*B' == SSM.gram_in_time(C)[:, :, 1]
        @test SSM.gram_in_time(C)[:, :, 2] == SSM.gram_in_time(C)[:, :, 1]
    end

    @testset "ensure_pos_sym" begin
        A = randn(3,3)
        B = SSM.ensure_pos_sym(A)
        @test issymmetric(B)
    end
end

function compare_forecast_simulation(ss::StateSpace, N::Int, S::Int, rtol::Float64)
    sim = simulate(ss, N, S)
    forec, dist = forecast(ss, N)

    @test forec ≈ mean(sim, dims = 3) rtol = rtol
end