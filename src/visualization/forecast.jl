@recipe function f(model::Union{StateSpaceModel, NaiveModel},
                   forec::Forecast)
    if !isunivariate(model)
        error("This plot recipe currently works for univariate models only.")
    end
    dists = Vector{Distribution}(undef, length(forec.expected_value))
    for d in eachindex(dists)
        dists[d] = Normal(forec.expected_value[d][1], sqrt(forec.covariance[d][1]))
    end
    n = num_observations(model)
    forec_idx = collect(n + 1: n + length(forec.expected_value))
    fillrange_color = :steelblue
    # Plot the series
    @series begin
        seriestype := :path
        seriescolor := "black"
        return observations(model)
    end
    # Plot the forecast
    @series begin
        seriescolor := "blue"
        return forec_idx, mean.(dists)
    end
    # Plot the prediction interval
    @series begin
        seriestype := :path
        seriescolor := fillrange_color
        fillcolor := fillrange_color
        fillalpha := 0.5
        fillrange := quantile.(dists, 0.975)
        label := ""
        return forec_idx, quantile.(dists, 0.025)
    end
    @series begin
        seriestype := :path
        seriescolor := fillrange_color
        fillcolor := fillrange_color
        fillalpha := 0.5
        fillrange := quantile.(dists, 0.90)
        label := ""
        return forec_idx, quantile.(dists, 0.10)
    end
end

RecipesBase.@recipe function f(model::Union{StateSpaceModel, NaiveModel},
                               scenarios::Array{<:AbstractFloat, 3})
    if !isunivariate(model)
        error("This plot recipe currently works for univariate models only.")
    end
    n = num_observations(model)
    n_scen = size(scenarios, 3)
    scenarios_idx = collect(n + 1: n + size(scenarios, 1))
    # Plot the series
    @series begin
        seriestype := :path
        seriescolor := "black"
        return observations(model)
    end
    # Plot the scenarios
    labels = ["" for s in 1:n_scen - 1]
    pushfirst!(labels, "scenarios")
    @series begin
        seriestype := :path
        seriescolor := "grey"
        linewidth := 0.2
        label := permutedims(labels)
        return scenarios_idx, scenarios[:, 1, :]
    end
end