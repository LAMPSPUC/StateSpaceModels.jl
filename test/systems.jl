@testset "Linear systems" begin
    @testset "LinearUnivariateTimeInvariant" begin
        y = rand(100)
        Z = rand(3)
        d = rand()
        T = rand(3, 3)
        c = rand(3)
        R = rand(3, 2)
        H = rand()
        Q = rand(2, 2)
        # Test if it throws an error with an invalid system matrices
        @test_throws MethodError LinearUnivariateTimeInvariant{Float64}(y, Z, T, R, d, c[1], H, Q)
        @test_throws MethodError LinearUnivariateTimeInvariant{Float64}(y, Z, T, R, d, c, H[:, :], Q)
        @test_throws MethodError LinearUnivariateTimeInvariant{Float64}(y, Z, T, R[1], d, c, H, Q)
        @test_throws DimensionMismatch LinearUnivariateTimeInvariant{Float64}(y, Z, T[:, 1:2], R, d, c, H, Q)
        @test_throws DimensionMismatch LinearUnivariateTimeInvariant{Float64}(y, Z[1:2], T, R, d, c, H, Q)
        @test_throws DimensionMismatch LinearUnivariateTimeInvariant{Float64}(y, Z, T, R, d, c, H, Q[:, 1:1])
    end

    @testset "LinearUnivariateTimeVariant" begin
        
    end

    @testset "LinearMultivariateTimeInvariant" begin
        
    end

    @testset "LinearMultivariateTimeVariant" begin
        
    end
end