@testset "User defined model" begin
    y = [1 2; 1 2.0]
    Z = [1 2.0; 2.0 1.0]
    T = [1 2.0]
    R = [1 2.0]
    @test_throws ErrorException StateSpaceModel(y, Z, T, R)

    Z = Array{Float64, 3}(undef, 2, 2, 2)
    Z[:, :, 1] = [1 2.0; 2.0 1.0]
    Z[:, :, 2] = [1 2.0; 2.0 1.0]
    T = [1 2.0; 2.0 1.0]
    R = [1 2.0; 2.0 1.0]
    ss = StateSpaceModel(y, Z, T, R)
    @test ss.mode == "time-variant"

    Z = Array{Float64, 3}(undef, 2, 2, 2)
    Z[:, :, 1] = [1 2.0; 2.0 1.0]
    Z[:, :, 2] = [1 2.0; 2.0 1.0]
    T = [1 2.0]
    R = [1 2.0; 2.0 1.0]
    @test_throws ErrorException StateSpaceModel(y, Z, T, R)
end