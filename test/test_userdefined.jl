@testset "User defined model" begin
    y = [1 2; 1 2.0]
    Z = Vector{Matrix{Float64}}(undef, 2)
    Z[1] = [1 2.0]
    Z[2] = [1 2.0]
    T = [1 2.0]
    R = [1 2.0]
    @test_throws ErrorException StateSpaceModel(y, Z, T, R)

    y = [1 2.0]
    Z = [1 2.0]
    T = [1 2.0]
    R = [1 2.0]
    @test_throws ErrorException StateSpaceModel(y, Z, T, R)
end