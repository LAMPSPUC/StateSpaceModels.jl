using StateSpaceModels, Plots

y = collect(1:0.25:20) + 0.5*randn(77)
y[10:20] .= NaN
model = linear_trend(y)
ss = statespace(model)

plot([y y], legend = :topleft, label = "observations", lw = 2, layout = (2, 1), xticks = 0:10:77, color = "black", grid = false)
plot!([ss.filter.a[:, 1] ss.smoother.alpha[:, 1]], label = ["filtered state" "smoothed state"], 
        color = ["indianred" "green"], lw = 2)