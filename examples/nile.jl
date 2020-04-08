using CSV, StateSpaceModels, Plots, Dates

# Load the Nile annual flow dataset
nile = CSV.read("Nile.csv")

# Convert data to an array of Float64
flow = Float64.(nile.Flow)

# Plot the data
p1 = plot(nile.Year, flow, label = "Annual flow", legend = :topright, color = :black)

# Specify the state-space model
model = local_level(flow)

# Estimate the state-space model
ss = statespace(model)

# Innovations
p2 = plot(nile.Year, ss.filter.v, label = "Innovations (v_t)", legend = :topright, color = :black)

# Plot filtered state and 90% confidence interval
std_ptt = sqrt.(ss.filter.Ptt[1, 1, :])
att = vec(ss.filter.att)
lb = vec(att + 1.64 * std_ptt)
ub = vec(att - 1.64 * std_ptt)
p3 = plot!(p1, nile.Year, att, ribbon = [ub - att, lb + att], 
           label = "Filtered state (att)", color = :red, fillalpha = 0.1)




