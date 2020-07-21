read_csv(file::String) = DataFrame!(CSV.File(file))

function monotone_forecast_variance(forec::StateSpaceModels.Forecast)
    for i in 2:length(forec.covariance)
        if all(diag(forec.covariance[i]) .<= diag(forec.covariance[i-1]))
            return false
        end
    end
    return true
end