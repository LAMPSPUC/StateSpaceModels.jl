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
        @test A == StateSpaceModels.gram(A)
        B = [1 1.0;0 1]
        @test B*B' == StateSpaceModels.gram(B)
    end

    @testset "gram_in_time" begin
        C = Array{Float64, 3}(undef, 2, 2, 5)
        B = [1 1.0;0 1]
        C[:, :, 1:5] .= B
        @test B*B' == StateSpaceModels.gram_in_time(C)[:, :, 1]
        @test StateSpaceModels.gram_in_time(C)[:, :, 2] == StateSpaceModels.gram_in_time(C)[:, :, 1]
    end
end