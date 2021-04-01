@recipe function f(model::Union{UnobservedComponents, 
                                ExponentialSmoothing},
                  kf::FilterOutput{<:AbstractFloat})
    if !isunivariate(model)
        error("This plot recipe currently works for univariate models only.")
    end
    components = dict_components(model)
    att = get_filtered_state(kf)
    # Plot the series
    layout := (length(components) + 1, 1)
    sp = 1 # subplot
    @series begin
        seriestype := :path
        label := ""
        seriescolor := "black"
        title := "Observed"
        subplot := sp
        model.system.y
    end
    sp += 1
    for (component, idx) in components
        @series begin
            seriestype := :path
            label := ""
            seriescolor := "black"
            title := "Filtered " * component
            subplot := sp
            att[:, idx]
        end
        sp += 1
    end
end

@recipe function f(model::Union{UnobservedComponents, 
                                ExponentialSmoothing}, 
                   ks::SmootherOutput{<:AbstractFloat})
    if !isunivariate(model)
        error("This plot recipe currently works for univariate models only.")
    end
    components = dict_components(model)
    alpha = get_smoothed_state(ks)
    # Plot the series
    layout := (length(components) + 1, 1)
    sp = 1 # subplot
    @series begin
        seriestype := :path
        label := ""
        seriescolor := "black"
        title := "Observed"
        subplot := sp
        model.system.y
    end
    sp += 1
    for (component, idx) in components
        @series begin
            seriestype := :path
            label := ""
            seriescolor := "black"
            title := "Smoothed " * component
            subplot := sp
            alpha[:, idx]
        end
        sp += 1
    end
end