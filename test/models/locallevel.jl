@testset "LocalLevel" begin 
    Nile_dataset = readdlm(joinpath(dirname(@__DIR__()), "datasets/Nile.csv"), ',')
    y = float.(Nile_dataset[2:end, 2])
    model = StateSpaceModels.LocalLevel(y)
    llk = StateSpaceModels.fit(model)
    @test StateSpaceModels.loglike(model) ≈ -640.98974 atol = 1e-5 rtol = 1e-5
end
