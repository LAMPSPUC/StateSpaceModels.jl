mutable struct Unknowns
    n_unknowns::Int
    unknown_indexes::Dict{String, Vector{CartesianIndex}}

    function Unknowns(ssm::StateSpaceModel)
        dict = Dict{String, Vector{CartesianIndex}}()
        dict["Z"] = find_nans_Z(ssm.Z)
        dict["T"] = find_nans(ssm.T)
        dict["R"] = find_nans(ssm.R)
        dict["d"] = find_nans(ssm.d)
        dict["c"] = find_nans(ssm.c)
        dict["sqrtH"] = find_nans_cov(ssm.H)
        dict["sqrtQ"] = find_nans_cov(ssm.Q)

        n_unknowns = num_unknowns(dict)

        if !isempty(dict["d"]) || !isempty(dict["c"])
            error("StateSpaceModels does not currently support estimating parameters in neither c and d.")
        end

        return new(n_unknowns, dict)
    end
end

function find_nans(mat::Matrix{T}) where T
    return findall(isnan, mat)
end

function find_nans_Z(Z::Array{T, 3}) where T
    t_dim = size(Z, 3)
    cart_idxs = Vector{Vector{CartesianIndex}}(undef, t_dim)
    for i in 1:t_dim
        cart_idxs[i] = findall(isnan, Z[:, :, i])
    end

    # Guarantee that the unknown is the same across all t
    @assert all_indexes_are_equal(cart_idxs)

    return cart_idxs[1]
end

function find_nans_cov(cov::Matrix{T}) where T
    return findall(isnan, tril(cov))
end

function all_indexes_are_equal(cart_vec::Vector{Vector{CartesianIndex}})
    for i in eachindex(cart_vec)
        if cart_vec[i] != cart_vec[1]
            return false
        end
    end
    return true
end

function num_unknowns(dict::Dict{String, Vector{CartesianIndex}})
    n = 0
    for (k, v) in dict
        n += length(v)
    end
    return n
end

function fill_model_with_parameters!(model::StateSpaceModel{Typ}, psitilde::Vector{Typ}, unknowns::Unknowns) where Typ
    # The order to fill the model is Z => T => R => sqrtH => sqrtQ
    offset = 0

    # Fill Z
    for un_Z in unknowns.unknown_indexes["Z"]
        offset += 1
        for t in axes(model.Z, 3)
            model.Z[un_Z, t] = psitilde[offset]
        end
    end

    # Fill T
    for un_T in unknowns.unknown_indexes["T"]
        offset += 1
        model.T[un_T] = psitilde[offset]
    end

    # Fill R
    for un_R in unknowns.unknown_indexes["R"]
        offset += 1
        model.R[un_R] = psitilde[offset]
    end

    # Fill d
    for un_d in unknowns.unknown_indexes["d"]
        offset += 1
        model.d[un_d] = psitilde[offset]
    end

    # Fill c
    for un_c in unknowns.unknown_indexes["c"]
        offset += 1
        model.c[un_c] = psitilde[offset]
    end

    # Fill sqrtH in the H matrix and do a gram (XX') after
    for un_H in unknowns.unknown_indexes["sqrtH"]
        offset += 1
        model.H[un_H] = psitilde[offset]
    end
    !isempty(unknowns.unknown_indexes["sqrtH"]) && fill_H!(model, gram(tril(model.H)))

    # Fill sqrtQ in the Q matrix and do a gram (XX') after
    for un_Q in unknowns.unknown_indexes["sqrtQ"]
        offset += 1
        model.Q[un_Q] = psitilde[offset]
    end
    !isempty(unknowns.unknown_indexes["sqrtQ"]) && fill_Q!(model, gram(tril(model.Q)))

    return 
end

function fill_H!(model::StateSpaceModel{T}, H::Matrix{T}) where T
    for i in axes(H, 1), j in axes(H, 2)
        model.H[i, j] = H[i, j]
    end
end

function fill_Q!(model::StateSpaceModel{T}, Q::Matrix{T}) where T
    for i in axes(Q, 1), j in axes(Q, 2)
        model.Q[i, j] = Q[i, j]
    end
end