@recipe function f(cv::CrossValidation, name::String)
    xguide := "lead times"
    @series begin
        seriestype := :path
        label := "MAE " * name
        marker := :circle
        cv.mae
    end
    @series begin
        seriestype := :path
        label := "Mean CRPS " * name
        marker := :circle
        cv.mean_crps
    end
end