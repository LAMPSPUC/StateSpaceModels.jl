@testset "User defined models" begin
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
    ssm = StateSpaceModel(y, Z, T, R)
    @test ssm.mode == "time-variant"

    Z = Array{Float64, 3}(undef, 2, 2, 2)
    Z[:, :, 1] = [1 2.0; 2.0 1.0]
    Z[:, :, 2] = [1 2.0; 2.0 1.0]
    T = [1 2.0]
    R = [1 2.0; 2.0 1.0]
    @test_throws ErrorException StateSpaceModel(y, Z, T, R)

    y = ones(5, 1)
    Z = ones(1, 3)
    T = ones(3, 3)
    R = ones(3, 3)
    d = zeros(5, 1) # correct d dimension
    c = zeros(5, 3) # correct c dimension
    H = SSM.build_H(1, Float64)
    Q = SSM.build_Q(3, 1, Float64)
    ssm = StateSpaceModel(y, Z, T, R, d, c, H, Q)
    @test ssm.dim == StateSpaceDimensions(5, 1, 3, 3)

    d = zeros(4, 1) # incorrect d dimension
    @test_throws AssertionError StateSpaceModel(y, Z, T, R, d, c, H, Q)

    d = zeros(5, 1) # correct d dimension
    c = zeros(5, 2) # incorrect c dimension
    @test_throws AssertionError StateSpaceModel(y, Z, T, R, d, c, H, Q)
end
