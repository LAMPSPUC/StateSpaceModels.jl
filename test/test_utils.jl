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

    @testset "ensure_pos_sym!" begin
        A = randn(3, 3, 2)
        SSM.ensure_pos_sym!(A, 1)
        @test issymmetric(A[:, :, 1])
        SSM.ensure_pos_sym!(A, 2)
        @test issymmetric(A[:, :, 2])
    end

    @testset "show" begin
        y = ones(15)
        model = local_level(y)
        ss = statespace(model)
        @test show(model) == nothing
        @test show(ss) == nothing
    end

    @testset "statespace_recursion" begin
        # Unknowns error
        y = ones(15)
        model = local_level(y)
        initial_a = [1.0][:, :]
        @test_throws ErrorException("StateSpaceModel has unknown parameters.") statespace_recursion(model, initial_a)

        # initial_a dimension mismatch
        y = ones(15)
        model = local_level(y)
        model.H .= [1e-8]
        model.Q .= [1e-8]
        initial_a = [1.0 1.0]
        @test_throws ErrorException("intial_a must be a 1 by 1 matrix.") statespace_recursion(model, initial_a)

        # univariate local_level
        y = ones(15)
        model = local_level(y)
        model.H .= [1e-8]
        model.Q .= [1e-8]
        initial_a = [1.0][:, :]
        y, a = statespace_recursion(model, initial_a)
        @test y ≈ ones(15) atol = 1e-3 rtol = 1e-3
        @test a ≈ ones(15) atol = 1e-3 rtol = 1e-3

        # univariate linear_trend
        y = collect(1.0:15.0)
        model = linear_trend(y)
        model.H .= [1e-8]
        model.Q .= [1e-8 0
                    0 1e-8]
        initial_a = [1.0 0.5]
        y, a = statespace_recursion(model, initial_a)
        @test y ≈ collect(1.0:0.5:8.0) atol = 1e-3 rtol = 1e-3
        @test a ≈ [collect(1.0:0.5:8.0) 0.5*ones(15)] atol = 1e-3 rtol = 1e-3

        # multivariate local_level
        y = ones(15, 4)
        model = local_level(y)
        model.H .= Diagonal(1e-8*ones(4))
        model.Q .= Diagonal(1e-8*ones(4))
        initial_a = ones(1, 4)
        y, a = statespace_recursion(model, initial_a)
        @test y ≈ ones(15, 4) atol = 1e-3 rtol = 1e-3
        @test a ≈ ones(15, 4) atol = 1e-3 rtol = 1e-3

        # univariate linear_trend
        y_c = collect(1.0:15.0)
        y = [y_c y_c y_c y_c]
        model = linear_trend(y)
        model.H .= Diagonal(1e-8*ones(4))
        model.Q .= Diagonal(1e-8*ones(8))
        initial_a = [1.0 0.5 1.0 0.5 1.0 0.5 1.0 0.5]
        y, a = statespace_recursion(model, initial_a)
        @test y[:, 1] ≈ collect(1.0:0.5:8.0) atol = 1e-3 rtol = 1e-3
        @test y[:, 2] ≈ collect(1.0:0.5:8.0) atol = 1e-3 rtol = 1e-3
        @test y[:, 3] ≈ collect(1.0:0.5:8.0) atol = 1e-3 rtol = 1e-3
        @test y[:, 4] ≈ collect(1.0:0.5:8.0) atol = 1e-3 rtol = 1e-3
        @test a[:, [1, 2]] ≈ [collect(1.0:0.5:8.0) 0.5*ones(15)] atol = 1e-3 rtol = 1e-3
        @test a[:, [3, 4]] ≈ [collect(1.0:0.5:8.0) 0.5*ones(15)] atol = 1e-3 rtol = 1e-3
        @test a[:, [5, 6]] ≈ [collect(1.0:0.5:8.0) 0.5*ones(15)] atol = 1e-3 rtol = 1e-3
        @test a[:, [7, 8]] ≈ [collect(1.0:0.5:8.0) 0.5*ones(15)] atol = 1e-3 rtol = 1e-3
    end
end

function compare_forecast_simulation(ss::StateSpace, N::Int, S::Int, rtol::Float64)
    sim = simulate(ss, N, S)
    forec, dist = forecast(ss, N)

    @test forec ≈ mean(sim, dims = 3) rtol = rtol
end