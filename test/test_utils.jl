@testset "utils.jl" begin

    @test "size_function" begin
        y = ones(15)
        X = randn(15, 2)
        s = 2
        model = structuralmodel(y, s; X = X)
        n, p, m, r = size(model)
        @test n == 15
        @test p == 1
        @test n == s + size(X, 2) + 1
        @test n == 3

        y = ones(15)
        X = randn(15, 15)
        s = 2
        model = structuralmodel(y, s; X = X)
        n, p, m, r = size(model)
        @test n == 15
        @test p == 1
        @test n == s + size(X, 2) + 1
        @test n == 3
    end

    @test "ztr" begin
        y = ones(15)
        X = randn(15, 2)
        s = 2
        model = structuralmodel(y, s; X = X)
        Z, T, R = ztr(model)

        #TODO This is easier to test when we have a local level model
    end

end