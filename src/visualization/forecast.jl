RecipesBase.@recipe function f(model::StateSpaceModel, forec::Forecast)
    if !isunivariate(model)
        error("This plot recipe currently works for univariate models only.")
    end
    dists = Vector{Distribution}(undef, length(forec.expected_value))
    for d in eachindex(dists)
        dists[d] = Normal(forec.expected_value[d][1], sqrt(forec.covariance[d][1]))
    end
    n = length(model.system.y)
    forec_idx = collect(n + 1: n + length(forec.expected_value))
    fillrange_color = :steelblue
    # Plot the series
    @series begin
        seriestype := :path
        seriescolor := "black"
        return model.system.y
    end
    # Plot the forecast
    @series begin
        seriescolor := "blue"
        return forec_idx, mean.(dists)
    end
    # Plot the Conf Interval
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