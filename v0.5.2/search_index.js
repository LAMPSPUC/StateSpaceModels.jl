var documenterSearchIndex = {"docs":
[{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"In this section we show examples of applications and use cases of the package.","category":"page"},{"location":"examples/#Nile-river-annual-flow","page":"Examples","title":"Nile river annual flow","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Here we will follow an example from Durbin & Koopman's book. We will use the LocalLevel model applied to the annual flow of the Nile river at the city of Aswan between 1871 and 1970.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using StateSpaceModels\nusing Plots\nusing Dates","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"First, we load the data:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using CSV, DataFrames\nnile = CSV.read(StateSpaceModels.NILE, DataFrame)\nplt = plot(nile.year, nile.flow, label = \"Nile river annual flow\")","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Next, we fit a LocalLevel model:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"model = LocalLevel(nile.flow)\nfit!(model)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"We can analyze the filtered estimates for the level of the annual flow:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"filter_output = kalman_filter(model)\nplot!(plt, nile.year, get_filtered_state(filter_output), label = \"Filtered level\")","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"We can do the same for the smoothed estimates for the level of the annual flow:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"smoother_output = kalman_smoother(model)\nplot!(plt, nile.year, get_smoothed_state(smoother_output), label = \"Smoothed level\")","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"StateSpaceModels.jl can also be used to obtain forecasts.  Here we forecast 10 steps ahead:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"steps_ahead = 10\ndates = collect(nile.year[end] + Year(1):Year(1):nile.year[end] + Year(10))\nforec = forecast(model, 10)\nexpected_value = forecast_expected_value(forec)\nplot!(plt, dates, expected_value, label = \"Forecast\")","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"We can also simulate multiple scenarios for the forecasting horizon based on the estimated predictive distributions.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"scenarios = simulate_scenarios(model, 10, 100)\nplot!(plt, dates, scenarios[:, 1, :], label = \"\", color = \"grey\", width = 0.2)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"The package also handles missing values automatically.  To that end, the package considers that any NaN entries in the observations are missing values.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"nile.flow[[collect(21:40); collect(61:80)]] .= NaN\nplt = plot(nile.year, nile.flow, label = \"Annual nile river flow\")","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Even though the series has several missing values, the same analysis is possible:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"model = LocalLevel(nile.flow)\nfit!(model)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"And the exact same code can be used for filtering and smoothing:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"filter_output = kalman_filter(model)\nsmoother_output = kalman_smoother(model)\nplot!(plt, nile.year, get_filtered_state(filter_output), label = \"Filtered level\")\nplot!(plt, nile.year, get_smoothed_state(smoother_output), label = \"Smoothed level\")","category":"page"},{"location":"examples/#Airline-passengers","page":"Examples","title":"Airline passengers","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"TODO","category":"page"},{"location":"examples/#Finland-road-traffic-fatalities","page":"Examples","title":"Finland road traffic fatalities","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"In this example, we will follow what is illustrated on Commandeur, Jacques J.F. & Koopman,  Siem Jan, 2007. \"An Introduction to State Space Time Series Analysis,\" OUP Catalogue,  Oxford University Press (Chapter 3). We will study the LocalLinearTrend model with  a series of the log of road traffic fatalities in Finalnd and analyse its slope to tell if the trend of fatalities was increasing or decrasing during different periods of time.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using StateSpaceModels\nusing Plots","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using CSV, DataFrames\ndf = CSV.read(StateSpaceModels.VEHICLE_FATALITIES, DataFrame)\nlog_ff = log.(df.ff)\nplt = plot(df.date, log_ff, label = \"Log of Finland road traffic fatalities\")","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"We fit a LocalLinearTrend","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"model = LocalLinearTrend(log_ff)\nfit!(model)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"By extracting the smoothed slope we conclude that according to our model the trend of fatalities  in Finland was increasing in the years 1970, 1982, 1984 through to 1988, and in 1998","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"smoother_output = kalman_smoother(model)\nplot(df.date, get_smoothed_state(smoother_output)[:, 2], label = \"slope\")","category":"page"},{"location":"manual/#Manual","page":"Manual","title":"Manual","text":"","category":"section"},{"location":"manual/#Models","page":"Manual","title":"Models","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"The package provides a variaty of pre-defined models. If there is any model that you wish was in the package, feel free to open an issue or pull request.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"ARIMA\nBasicStructural\nLinearRegression\nLocalLevel\nLocalLevelCycle\nLocalLevelExplanatory\nLocalLinearTrend\nMultivariateBasicStructural","category":"page"},{"location":"manual/#StateSpaceModels.ARIMA","page":"Manual","title":"StateSpaceModels.ARIMA","text":"ARIMA\n\nAn ARIMA model (autoregressive integrated moving average) implemented within the state-space framework.\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.BasicStructural","page":"Manual","title":"StateSpaceModels.BasicStructural","text":"The basic structural state-space model consists of a trend (level + slope) and a seasonal component. It is defined by:\n\nbegingather*\n    beginaligned\n        y_t =  mu_t + gamma_t + varepsilon_t quad varepsilon_t sim mathcalN(0 sigma^2_varepsilon)\n        mu_t+1 = mu_t + nu_t + xi_t quad xi_t sim mathcalN(0 sigma^2_xi)\n        nu_t+1 = nu_t + zeta_t quad zeta_t sim mathcalN(0 sigma^2_zeta)\n        gamma_t+1 = -sum_j=1^s-1 gamma_t+1-j + omega_t quad  omega_t sim mathcalN(0 sigma^2_omega)\n    endaligned\nendgather*\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.LinearRegression","page":"Manual","title":"StateSpaceModels.LinearRegression","text":"The linear regression state-space model is defined by:\n\nbegingather*\n    beginaligned\n        y_t =  X_1t cdot beta_1t + dots + X_nt cdot beta_nt + varepsilon_t quad varepsilon_t sim mathcalN(0 sigma^2_varepsilon)\n        beta_1t+1 = beta_1t\n        dots = dots\n        beta_nt+1 = beta_nt\n    endaligned\nendgather*\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.LocalLevel","page":"Manual","title":"StateSpaceModels.LocalLevel","text":"The local level model is defined by:\n\nbegingather*\n    beginaligned\n        y_t =  mu_t + varepsilon_t quad varepsilon_t sim mathcalN(0 sigma^2_varepsilon)\n        mu_t+1 = mu_t + eta_t quad eta_t sim mathcalN(0 sigma^2_eta)\n    endaligned\nendgather*\n\nExample\n\njulia> model = LocalLevel(rand(100))\nLocalLevel model\n\nSee more on Nile river annual flow\n\nReferences\n\nDurbin, James, & Siem Jan Koopman. (2012). \"Time Series Analysis by State Space Methods: Second Edition.\" Oxford University Press. pp. 9\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.LocalLevelCycle","page":"Manual","title":"StateSpaceModels.LocalLevelCycle","text":"The local level model with a cycle component is defined by:\n\nbegingather*\n    beginaligned\n        y_t =  mu_t + c_t + varepsilon_t quad varepsilon_t sim mathcalN(0 sigma^2_varepsilon)\n        mu_t+1 = mu_t + eta_t quad eta_t sim mathcalN(0 sigma^2_eta)\n        c_t+1 = c_t cos(lambda_c) + c_t^* sin(lambda_c) quad omega_1t sim mathcalN(0 sigma^2_omega_1)\n        c_t+1^* = -c_t sin(lambda_c) + c_t^* sin(lambda_c) quad omega_2t sim mathcalN(0 sigma^2_omega_2)\n    endaligned\nendgather*\n\nExample\n\njulia> model = LocalLevelCycle(rand(100))\nLocalLevelCycle model\n\nSee more on TODO RJ_TEMPERATURE\n\nReferences\n\nDurbin, James, & Siem Jan Koopman. (2012). \"Time Series Analysis by State Space Methods: Second Edition.\" Oxford University Press. pp. 48\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.LocalLevelExplanatory","page":"Manual","title":"StateSpaceModels.LocalLevelExplanatory","text":"A local level model with explanatory variables is defined by:\n\nbegingather*\n    beginaligned\n        y_t =  mu_t + X_1t cdot beta_1t + dots + X_nt cdot beta_nt + varepsilon_t quad varepsilon_t sim mathcalN(0 sigma^2_varepsilon)\n        mu_t+1 = mu_t + xi_t xi_t sim mathcalN(0 sigma^2_xi)\n        beta_1t+1 = beta_1t tau_1 t sim mathcalN(0 sigma^2_tau_1)\n        dots = dots\n        beta_nt+1 = beta_nt tau_n t sim mathcalN(0 sigma^2_tau_n)\n    endaligned\nendgather*\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.LocalLinearTrend","page":"Manual","title":"StateSpaceModels.LocalLinearTrend","text":"The linear trend model is defined by:\n\nbegingather*\n    beginaligned\n        y_t =  mu_t + gamma_t + varepsilon_t quad varepsilon_t sim mathcalN(0 sigma^2_varepsilon)\n        mu_t+1 = mu_t + nu_t + xi_t quad xi_t sim mathcalN(0 sigma^2_xi)\n        nu_t+1 = nu_t + zeta_t quad zeta_t sim mathcalN(0 sigma^2_zeta)\n    endaligned\nendgather*\n\nExample\n\njulia> model = LocalLinearTrend(rand(100))\nLocalLinearTrend model\n\nSee more on Finland road traffic fatalities\n\nReferences\n\nDurbin, James, & Siem Jan Koopman. (2012). \"Time Series Analysis by State Space Methods:  Second Edition.\" Oxford University Press. pp. 44\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.MultivariateBasicStructural","page":"Manual","title":"StateSpaceModels.MultivariateBasicStructural","text":"An implementation of a non-homogeneous seemingly unrelated time series equations for basic structural state-space model consists of trend (local linear trend) and seasonal components. It is defined by:\n\nbegingather*\n    beginaligned\n    y_t =  mu_t + gamma_t + varepsilon_t quad varepsilon_t sim mathcalN(0 Sigma_varepsilon)\n    mu_t+1 = mu_t + nu_t + xi_t quad xi_t sim mathcalN(0 Sigma_xi)\n    nu_t+1 = nu_t + zeta_t quad zeta_t sim mathcalN(0 Sigma_zeta)\n    gamma_t+1 = -sum_j=1^s-1 gamma_t+1-j + omega_t quad  omega_t sim mathcalN(0 Sigma_omega)\n    endaligned\nendgather*\n\n\n\n\n\n","category":"type"},{"location":"manual/#Implementing-a-custom-model","page":"Manual","title":"Implementing a custom model","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"Users are able to implement any custom user-defined model.","category":"page"},{"location":"manual/#Systems","page":"Manual","title":"Systems","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"The StateSpaceModel matrices are represented as a StateSpaceSystem.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"StateSpaceModels.StateSpaceSystem\nLinearUnivariateTimeInvariant\nLinearUnivariateTimeVariant\nLinearMultivariateTimeInvariant\nLinearMultivariateTimeVariant","category":"page"},{"location":"manual/#StateSpaceModels.StateSpaceSystem","page":"Manual","title":"StateSpaceModels.StateSpaceSystem","text":"StateSpaceSystem\n\nAbstract type that unifies the definition of state space models matrices such as y Z d T c R H Q for linear models.\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.LinearUnivariateTimeInvariant","page":"Manual","title":"StateSpaceModels.LinearUnivariateTimeInvariant","text":"LinearUnivariateTimeInvariant\n\nDefinition of the system matrices y Z d T c R H Q for linear univariate time invariant state space models.\n\nbegingather*\n    beginaligned\n        y_t =  Zalpha_t + d + varepsilon_t quad varepsilon_t sim mathcalN(0 H)\n        alpha_t+1 = Talpha_t + c + Reta_t quad eta_t sim mathcalN(0 Q)\n    endaligned\nendgather*\n\nwhere:\n\ny_t is a scalar\nZ is a m times 1 vector\nd is a scalar\nT is a m times m matrix\nc is a m times 1 vector\nR is a m times r matrix\nH is a scalar\nQ is a r times r matrix\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.LinearUnivariateTimeVariant","page":"Manual","title":"StateSpaceModels.LinearUnivariateTimeVariant","text":"LinearUnivariateTimeVariant\n\nDefinition of the system matrices y Z d T c R H Q for linear univariate time variant state space models.\n\nbegingather*\n    beginaligned\n        y_t =  Z_talpha_t + d_t + varepsilon_t quad varepsilon_t sim mathcalN(0 H_t)\n        alpha_t+1 = T_talpha_t + c_t + R_teta_t quad eta_t sim mathcalN(0 Q_t)\n    endaligned\nendgather*\n\nwhere:\n\ny_t is a scalar\nZ_t is a m times 1 vector\nd_t is a scalar\nT_t is a m times m matrix\nc_t is a m times 1 vector\nR_t is a m times r matrix\nH_t is a scalar\nQ_t is a r times r matrix\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.LinearMultivariateTimeInvariant","page":"Manual","title":"StateSpaceModels.LinearMultivariateTimeInvariant","text":"TODO\n\n\n\n\n\n","category":"type"},{"location":"manual/#StateSpaceModels.LinearMultivariateTimeVariant","page":"Manual","title":"StateSpaceModels.LinearMultivariateTimeVariant","text":"TODO\n\n\n\n\n\n","category":"type"},{"location":"manual/#Hyperparameters","page":"Manual","title":"Hyperparameters","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"The model hyperparameters are constant (non-time-varying) parameters that are optimized when fit! is called. The package provides some useful getters and setters to accelerate experimentation with models.","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"The getters are:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"get_hyperparameters\nget_names","category":"page"},{"location":"manual/#StateSpaceModels.get_names","page":"Manual","title":"StateSpaceModels.get_names","text":"get_names(model::StateSpaceModel)\n\nGet the names of the hyperparameters registered on a StateSpaceModel.\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"The setters are:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"fix_hyperparameters!","category":"page"},{"location":"manual/#StateSpaceModels.fix_hyperparameters!","page":"Manual","title":"StateSpaceModels.fix_hyperparameters!","text":"fix_hyperparameters!(model::StateSpaceModel, fixed_hyperparameters::Dict)\n\nFixes the desired hyperparameters so that they are not considered as decision variables in the model estimation.\n\nExample\n\njulia> model = LocalLevel(rand(100));\n\njulia> get_names(model)\n2-element Array{String,1}:\n \"sigma2_ε\"\n \"sigma2_η\"\n\njulia> fix_hyperparameters!(model, Dict(\"sigma2_ε\" => 100.0))\nStateSpaceModels.HyperParameters{Float64}(2, [\"sigma2_ε\", \"sigma2_η\"], Dict(\"sigma2_η\" => 2,\"sigma2_ε\" => 1), Dict{Int64,String}(), [NaN, NaN], [NaN, NaN], Dict(\"sigma2_ε\" => 100.0))\n\n\n\n\n\n","category":"function"},{"location":"manual/","page":"Manual","title":"Manual","text":"Mappings:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"constrain_variance!\nunconstrain_variance!\nconstrain_box!\nunconstrain_box!\nconstrain_identity!\nunconstrain_identity!","category":"page"},{"location":"manual/#StateSpaceModels.constrain_variance!","page":"Manual","title":"StateSpaceModels.constrain_variance!","text":"constrain_variance!(model::StateSpaceModel, str::String)\n\nMap a constrained hyperparameter psi in mathbbR^+ to an unconstrained hyperparameter psi_* in mathbbR.\n\nThe mapping is psi = psi_*^2\n\n\n\n\n\n","category":"function"},{"location":"manual/#StateSpaceModels.unconstrain_variance!","page":"Manual","title":"StateSpaceModels.unconstrain_variance!","text":"unconstrain_variance!(model::StateSpaceModel, str::String)\n\nMap an unconstrained hyperparameter psi_* in mathbbR to a constrained hyperparameter  psi in mathbbR^+.\n\nThe mapping is psi_* = sqrtpsi.\n\n\n\n\n\n","category":"function"},{"location":"manual/#StateSpaceModels.constrain_box!","page":"Manual","title":"StateSpaceModels.constrain_box!","text":"constrain_box!(model::StateSpaceModel, str::String, lb::Fl, ub::Fl) where Fl\n\nMap a constrained hyperparameter psi in lb ub to an unconstrained hyperparameter psi_* in mathbbR.\n\nThe mapping is psi = lb + fracub - lb1 + exp(-psi_*)\n\n\n\n\n\n","category":"function"},{"location":"manual/#StateSpaceModels.unconstrain_box!","page":"Manual","title":"StateSpaceModels.unconstrain_box!","text":"unconstrain_box!(model::StateSpaceModel, str::String, lb::Fl, ub::Fl) where Fl\n\nMap an unconstrained hyperparameter psi_* in mathbbR to a constrained hyperparameter psi in lb ub.\n\nThe mapping is psi_* = -ln fracub - lbpsi - lb - 1\n\n\n\n\n\n","category":"function"},{"location":"manual/#StateSpaceModels.constrain_identity!","page":"Manual","title":"StateSpaceModels.constrain_identity!","text":"constrain_identity!(model::StateSpaceModel, str::String)\n\nMap an constrained hyperparameter psi in mathbbR to an unconstrained hyperparameter psi_* in mathbbR. This function is necessary to copy values from a location to another inside HyperParameters\n\nThe mapping is psi = psi_*\n\n\n\n\n\n","category":"function"},{"location":"manual/#StateSpaceModels.unconstrain_identity!","page":"Manual","title":"StateSpaceModels.unconstrain_identity!","text":"unconstrain_identity!(model::StateSpaceModel, str::String)\n\nMap an unconstrained hyperparameter psi_* in mathbbR to a constrained hyperparameter psi in mathbbR. This function is necessary to copy values from a location to another inside HyperParameters\n\nThe mapping is psi_* = psi\n\n\n\n\n\n","category":"function"},{"location":"manual/#Optim.jl-interface","page":"Manual","title":"Optim.jl interface","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"The optimizer to be used can be configured through an interface with Optim.jl:","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"Optimizer","category":"page"},{"location":"manual/#StateSpaceModels.Optimizer","page":"Manual","title":"StateSpaceModels.Optimizer","text":"Optimizer\n\nAn Optim.jl wrapper to make the choice of the optimizer straightforward in StateSpaceModels.jl Users can choose among all suitable Optimizers in Optim.jl using very similar syntax.\n\nExample\n\njulia> using Optim\n\n# use a semicolon to avoid displaying the big log\njulia> opt = Optimizer(Optim.LBFGS(), Optim.Options(show_trace = true));\n\n\n\n\n\n","category":"type"},{"location":"manual/#Datasets","page":"Manual","title":"Datasets","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"The package provides some datasets to illustrate the funtionalities and models.  These datasets are stored as csv files and the path to these files can be obtained through their names as seen below. In the examples we illustrate the datasets using DataFrames.jl and CSV.jl","category":"page"},{"location":"manual/","page":"Manual","title":"Manual","text":"StateSpaceModels.AIR_PASSENGERS\nStateSpaceModels.FRONT_REAR_SEAT_KSI\nStateSpaceModels.INTERNET\nStateSpaceModels.NILE\nStateSpaceModels.RJ_TEMPERATURE\nStateSpaceModels.VEHICLE_FATALITIES\nStateSpaceModels.WHOLESALE_PRICE_INDEX","category":"page"},{"location":"manual/#StateSpaceModels.AIR_PASSENGERS","page":"Manual","title":"StateSpaceModels.AIR_PASSENGERS","text":"AIR_PASSENGERS\n\nThe absolute path for the so called AIR_PASSENGERS dataset stored inside StateSpaceModels.jl. This dataset provides monthly totals of a US airline passengers from 1949 to 1960.\n\nSee more on Log of airline passengers\n\nReferences\n\nhttps://www.stata-press.com/data/r12/ts.html\n\n\n\n\n\n","category":"constant"},{"location":"manual/#StateSpaceModels.FRONT_REAR_SEAT_KSI","page":"Manual","title":"StateSpaceModels.FRONT_REAR_SEAT_KSI","text":"FRONT_REAR_SEAT_KSI\n\nThe absolute path for the so called FRONT_REAR_SEAT_KSI dataset stored inside StateSpaceModels.jl. This dataset provides the log of british people killed or serious injuried in road accidents accross UK.\n\nReferences\n\nCommandeur, Jacques J.F. & Koopman, Siem Jan, 2007. \"An Introduction to State Space Time  Series Analysis,\" OUP Catalogue, Oxford University Press (Chapter 3)\nhttp://staff.feweb.vu.nl/koopman/projects/ckbook/OxCodeAll.zip\n\n\n\n\n\n","category":"constant"},{"location":"manual/#StateSpaceModels.INTERNET","page":"Manual","title":"StateSpaceModels.INTERNET","text":"INTERNET\n\nThe absolute path for the so called INTERNET dataset stored inside StateSpaceModels.jl. This dataset provides the number of users logged on to an Internet server each minute over 100 minutes.\n\nSee more on ARIMA... under contruction\n\nReferences\n\nDurbin, James, & Siem Jan Koopman. (2012). \"Time Series Analysis by State Space Methods:  Second Edition.\" Oxford University Press. (Chapter 9)\n\n\n\n\n\n","category":"constant"},{"location":"manual/#StateSpaceModels.NILE","page":"Manual","title":"StateSpaceModels.NILE","text":"NILE\n\nThe absolute path for the so called NILE dataset stored inside StateSpaceModels.jl. This dataset provides measurements of the annual flow of the Nile river at Aswan from 1871 to 1970, in 10^8 m^3.\n\nSee more on Nile river annual flow\n\nReferences\n\nDurbin, James, & Siem Jan Koopman. (2012). \"Time Series Analysis by State Space Methods: Second Edition.\" Oxford University Press. (Chapter 2)\n\n\n\n\n\n","category":"constant"},{"location":"manual/#StateSpaceModels.RJ_TEMPERATURE","page":"Manual","title":"StateSpaceModels.RJ_TEMPERATURE","text":"TODO\n\n\n\n\n\n","category":"constant"},{"location":"manual/#StateSpaceModels.VEHICLE_FATALITIES","page":"Manual","title":"StateSpaceModels.VEHICLE_FATALITIES","text":"VEHICLE_FATALITIES\n\nThe absolute path for the so called VEHICLE_FATALITIES dataset stored inside StateSpaceModels.jl. This dataset provides the number of annual road traffic fatalities in Norway and Finland.\n\nSee more on Finland road traffic fatalities\n\nReferences\n\nCommandeur, Jacques J.F. & Koopman, Siem Jan, 2007. \"An Introduction to State Space Time  Series Analysis,\" OUP Catalogue, Oxford University Press (Chapter 3)\nhttp://staff.feweb.vu.nl/koopman/projects/ckbook/OxCodeAll.zip\n\n\n\n\n\n","category":"constant"},{"location":"manual/#StateSpaceModels.WHOLESALE_PRICE_INDEX","page":"Manual","title":"StateSpaceModels.WHOLESALE_PRICE_INDEX","text":"WHOLESALE_PRICE_INDEX\n\nThe absolute path for the so called INTERNET dataset stored inside StateSpaceModels.jl. This dataset provides the number of users logged on to an Internet server each minute over 100 minutes.\n\nSee more on ARIMA... under contruction\n\nReferences\n\nhttps://www.stata.com/manuals13/tsarima.pdf\n\n\n\n\n\n","category":"constant"},{"location":"","page":"Home","title":"Home","text":"<div style=\"width:100%; height:150px;border-width:4px;border-style:solid;padding-top:25px;\n        border-color:#000;border-radius:10px;text-align:center;background-color:#99DDFF;\n        color:#000\">\n    <h3 style=\"color: black;\">Star us on GitHub!</h3>\n    <a class=\"github-button\" href=\"https://github.com/LAMPSPUC/StateSpaceModels.jl\" data-icon=\"octicon-star\" data-size=\"large\" data-show-count=\"true\" aria-label=\"Star LAMPSPUC/StateSpaceModels.jl on GitHub\" style=\"margin:auto\">Star</a>\n    <script async defer src=\"https://buttons.github.io/buttons.js\"></script>\n</div>","category":"page"},{"location":"#StateSpaceModels.jl-Documentation","page":"Home","title":"StateSpaceModels.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"StateSpaceModels.jl is a package for modeling, forecasting, and simulating time series in a state-space framework. Implementations were made based on the book \"Time Series Analysis by State Space Methods\" (2012) by James Durbin and Siem Jan Koopman. The notation of the variables in the code also aims to follow the book.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package is registered so you can simply add it using Julia's Pkg manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add StateSpaceModels","category":"page"},{"location":"#Citing-StateSpaceModels.jl","page":"Home","title":"Citing StateSpaceModels.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you use StateSpaceModels.jl in your work, we kindly ask you to cite the following paper (pdf):","category":"page"},{"location":"","page":"Home","title":"Home","text":"@article{SaavedraBodinSouto2019,\ntitle={StateSpaceModels.jl: a Julia Package for Time-Series Analysis in a State-Space Framework},\nauthor={Raphael Saavedra and Guilherme Bodin and Mario Souto},\njournal={arXiv preprint arXiv:1908.01757},\nyear={2019}\n}","category":"page"}]
}
