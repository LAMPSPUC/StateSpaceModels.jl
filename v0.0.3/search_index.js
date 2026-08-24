var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#StateSpaceModels.jl-Documentation-1",
    "page": "Home",
    "title": "StateSpaceModels.jl Documentation",
    "category": "section",
    "text": ""
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "This package is unregistered so you will need to Pkg.add it as follows:Pkg.add(\"https://github.com/LAMPSPUC/StateSpaceModels.jl.git\")"
},

{
    "location": "#Notes-1",
    "page": "Home",
    "title": "Notes",
    "category": "section",
    "text": "StateSpaceModels.jl is a package for modeling, forecasting and simulating time series in a state space framework. Implementations were made based on the book Time series analysis by state space methods: J. Durbin and S.J. Koopman."
},

{
    "location": "#Features-1",
    "page": "Home",
    "title": "Features",
    "category": "section",
    "text": "Current features:Basic structural model (level, slope, seasonal)\nExogenous variables\nSquare-root Kalman Filter and smoother\nBig Kappa initialization\nMonte Carlo simulation\nParallel MLE estimation\nMultivariate modelingFuture features (work in progress):User-defined model\nCycles\nForecasting and confidence intervals\nCompletion of missing values\nDifferent models for each variable in multivariate case\nStructural break\nExact initialization"
},

{
    "location": "manual/#",
    "page": "Manual",
    "title": "Manual",
    "category": "page",
    "text": ""
},

{
    "location": "manual/#Manual-1",
    "page": "Manual",
    "title": "Manual",
    "category": "section",
    "text": ""
},

{
    "location": "manual/#Estimation-1",
    "page": "Manual",
    "title": "Estimation",
    "category": "section",
    "text": "The model estimation is made using the function statespace(y, s; X, nseeds). It receives as argument the timeseries and the desired seasonality s.The user can input explanatory variables in an Array{Float64, 2} variable X and specify the desired number of seeds to perform the estimation nseeds.ss = statespace(y, s; X = X, nseeds = nseeds)"
},

{
    "location": "manual/#Simulation-1",
    "page": "Manual",
    "title": "Simulation",
    "category": "section",
    "text": "Simulation is made using the function simulate. It receives as argument a StateSpace object, the number of steps ahead N and the number of scenarios to simulate S.simulation = simulate(ss, N, S)"
},

{
    "location": "manual/#Example-1",
    "page": "Manual",
    "title": "Example",
    "category": "section",
    "text": "Letś take the Air Passenger time series to build and example. Taking the log of the series we should have a nice timeseries to simulate. The code is in the example folder.using CSV, StateSpaceModels, Plots, Statistics, Dates\n\n#load the AirPassengers dataset\nAP = CSV.read(\"AirPassengers.csv\")\n\n#Take the log of the series\nlogAP = log.(Array{Float64}(AP[:Passengers]))\n\np1 = plot(AP[:Date], logAP, label = \"AirPassengers timeseries\", size = (1000, 500))(Image: Log of Air Passengers time series)Estimating a StateSpaceModel should give us the trend and seasonal components of the time series#Define its seasonality \ns = 12\n\n#Estimate a StateSpace Structure\nss = statespace(logAP, s)\n\n#Analyze its decomposition in seasonal and trend\np2 = plot(AP[:Date], ss.state.seasonal, label = \"AirPassengers seasonal\", size = (1000, 500))\np3 = plot(AP[:Date], ss.state.trend, label = \"AirPassengers trend\", size = (1000, 500))(Image: Lof of Air Passengers trend component) (Image: Log of Air Passengers seasonal component)We can also simulate the future of this time series. ?In this example we simulate 100 scenarios of 60 steps ahead.#Simulate 100 scenarios, 20 steps ahead\nnum_scenarios = 100\nnum_steps_ahead = 60\nsimulation = simulate(ss, num_steps_ahead, num_scenarios)\n\n#Define simulation dates\nfirstdate = AP[:Date][end] + Month(1)\nnewdates = collect(firstdate:Month(1):firstdate + Month(num_steps_ahead - 1))\n\n#Evaluating the mean of the forecast and its quantiles\nsimulation_mean = mean(simulation, dims = 3)[1, :, :]\n\nn = length(logAP)\nnmonths = length(simulation[1, :, 1])\nsimulation_q05 = zeros(nmonths)\nsimulation_q95 = zeros(nmonths)\nfor t = 1:nmonths\n    simulation_q05[t] = quantile(simulation[1, t, :], 0.05)\n    simulation_q95[t] = quantile(simulation[1, t, :], 0.95)\nend\n\nplot!(p1, newdates, [simulation_q05, simulation_mean, simulation_q95], labels = [\"5% quantile\", \"mean\", \"95% quantile\"])(Image: Log of Air Passengers simulation)"
},

]}
