@testset "Linear systems" begin
    @testset "LinearUnivariateTimeInvariant" begin
        y = rand(100)
        Z = ones(3)
        d = rand()
        T = Matrix(Diagonal(ones(3, 3)))
        c = zeros(3)
        R = [
            1.0 0.0
            0.0 1.0
            0.0 0.0
        ]
        H = rand()
        Q = Matrix(Diagonal(rand(2, 2)))
        # Test if it throws an error with an invalid system matrices
        @test_throws MethodError LinearUnivariateTimeInvariant{Float64}(
            y, Z, T, R, d, c[1], H, Q
        )
        @test_throws MethodError LinearUnivariateTimeInvariant{Float64}(
            y, Z, T, R, d, c, H[:, :], Q
        )
        @test_throws MethodError LinearUnivariateTimeInvariant{Float64}(
            y, Z, T, R[1], d, c, H, Q
        )
        @test_throws DimensionMismatch LinearUnivariateTimeInvariant{Float64}(
            y, Z, T[:, 1:2], R, d, c, H, Q
        )
        @test_throws DimensionMismatch LinearUnivariateTimeInvariant{Float64}(
            y, Z[1:2], T, R, d, c, H, Q
        )
        @test_throws DimensionMismatch LinearUnivariateTimeInvariant{Float64}(
            y, Z, T, R, d, c, H, Q[:, 1:1]
        )

        system = StateSpaceModels.LinearUnivariateTimeInvariant{Float64}(
            y, Z, T, R, d, c, H, Q
        )

        n_obs_sim = 200
        initial_state = [0.0; 0.0; 0.7]
        sim = StateSpaceModels.simulate(system, initial_state, n_obs_sim)
        @test length(sim) == 200
        @test maximum(sim) <= 1e5 # This should not explode

        sim, alpha = StateSpaceModels.simulate(
            system, initial_state, n_obs_sim; return_simulated_states=true
        )
        @test size(alpha) == (200, 3)
        # The third state has 0.0 variance and initial_state 0.7 so this should not change
        @test all(alpha[:, 3] .== 0.7)
    end

    @testset "LinearUnivariateTimeVariant" begin
        # TODO
    end

    @testset "LinearMultivariateTimeInvariant" begin
        # TODO
    end

    @testset "LinearMultivariateTimeVariant" begin
        # TODO
    end
end
