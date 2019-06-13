export size, ztr

"""
    size(model::StateSpaceModel)

Return the dimensions `n`, `p`, `m` and `r` of the `StateSpaceModel`
"""
function size(model::StateSpaceModel)
    return model.dim.n, model.dim.p, model.dim.m, model.dim.r
end

"""
    matrices(model::StateSpaceModel)

Return the state space model arrays `Z`, `T` and `R` of the `StateSpaceModel`
"""
function ztr(model::StateSpaceModel)
    return model.Z, model.T, model.R
end


# Linar Algebra wrappers
function gram_in_time(mat::Array{Float64, 3})
    gram_in_time = similar(mat)
    for t = 1:size(gram_mat, 3)
        gram_in_time[:, :, t] = gram(mat)
    end
    return gram_in_time
end

function gram(mat::Matrix{T}) where T <: AbstractFloat
    return mat*mat'    
end