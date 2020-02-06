using CSV, StateSpaceModels, Plots, Statistics, Dates

# Load the AirPassengers dataset
AP = CSV.read("AirPassengers.csv")

# Take the log of the series
logAP = log.(Vector{Float64}(AP.Passengers))

# Plot the data
p1 = plot(AP.Date, logAP, label = "Log-airline passengers", legend = :topleft, color = :black)

# Specify the state-space model
model = structural(logAP, 12)

# Estimate the state-space model
ss = statespace(model)

# Analyze its decomposition in level, trend and seasonal
p2 = plot(AP.Date, [ss.smoother.alpha[:, 1] ss.smoother.alpha[:, 2] ss.smoother.alpha[:, 3]], layout = (3, 1),
            label = ["Level component" "Trend component" "Seasonal component"], legend = :topleft)

# Forecast 24 months ahead
N = 24
pred, dist = forecast(ss, N)

# Define forecasting dates
firstdate = AP.Date[end] + Month(1)
newdates = collect(firstdate:Month(1):firstdate + Month(N - 1))

p3 = plot!(p1, newdates, pred, label = "Forecast")

# 10 scenarios for 24 months ahead
sim = simulate(ss, N, 10)
p4 = plot!(p1, newdates, sim[:, 1, :], label = "", color = "Grey", width = 0.2)