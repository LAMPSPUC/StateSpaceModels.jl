# Datasets are not exported to avoid colliding with names in the Main scope

@doc raw"""
    NILE

The absolute path for the `NILE` dataset stored inside StateSpaceModels.jl.
This dataset provides measurements of the annual flow of the Nile river at Aswan from 1871 to 1970,
in ``10^8 m^3``.

See more on [Nile river annual flow](@ref)

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. (Chapter 2)
"""
const NILE = joinpath(dirname(@__DIR__()), "datasets", "nile.csv")

@doc raw"""
    AIR_PASSENGERS

The absolute path for the `AIR_PASSENGERS` dataset stored inside StateSpaceModels.jl.
This dataset provides monthly totals of a US airline passengers from 1949 to 1960.

See more on [Airline passengers](@ref)

# References
 * https://www.stata-press.com/data/r12/ts.html
"""
const AIR_PASSENGERS = joinpath(dirname(@__DIR__()), "datasets", "airpassengers.csv")

@doc raw"""
    INTERNET

The absolute path for the `INTERNET` dataset stored inside StateSpaceModels.jl.
This dataset provides the number of users logged on to an Internet server each minute over 100 minutes.

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods:
    Second Edition." Oxford University Press. (Chapter 9)
"""
const INTERNET = joinpath(dirname(@__DIR__()), "datasets", "internet.csv")

@doc raw"""
    WHOLESALE_PRICE_INDEX

TODO
"""
const WHOLESALE_PRICE_INDEX = joinpath(
    dirname(@__DIR__()), "datasets", "wholesalepriceindex.csv"
)

@doc raw"""
    VEHICLE_FATALITIES

The absolute path for the `VEHICLE_FATALITIES` dataset stored inside StateSpaceModels.jl.
This dataset provides the number of annual road traffic fatalities in Norway and Finland.

See more on [Finland road traffic fatalities](@ref)

# References
 * Commandeur, Jacques J.F. & Koopman, Siem Jan, 2007. "An Introduction to State Space Time
    Series Analysis," OUP Catalogue, Oxford University Press (Chapter 3)
 * http://staff.feweb.vu.nl/koopman/projects/ckbook/OxCodeAll.zip
"""
const VEHICLE_FATALITIES = joinpath(
    dirname(@__DIR__()), "datasets", "vehicle_fatalities.csv"
)

@doc raw"""
    FRONT_REAR_SEAT_KSI

The absolute path for the `FRONT_REAR_SEAT_KSI` dataset stored inside StateSpaceModels.jl.
This dataset provides the log of british people killed or serious injuried in road accidents accross
UK.

# References
 * Commandeur, Jacques J.F. & Koopman, Siem Jan, 2007. "An Introduction to State Space Time
    Series Analysis," OUP Catalogue, Oxford University Press (Chapter 3)
 * http://staff.feweb.vu.nl/koopman/projects/ckbook/OxCodeAll.zip
"""
const FRONT_REAR_SEAT_KSI = joinpath(dirname(@__DIR__()), "datasets", "front_rear_seat.csv")

@doc raw"""
    RJ_TEMPERATURE

Weekly mean temperature in Rio de Janeiro in Kelvin (K).
"""
const RJ_TEMPERATURE = joinpath(dirname(@__DIR__()), "datasets", "rj_temperature.csv")

@doc raw"""
    US_CHANGE

Percentage changes in quarterly personal consumption expenditure, personal disposable income,
production, savings and the unemployment rate for the US, 1960 to 2016.

Federal Reserve Bank of St Louis.

# References
 * Hyndman, Rob J., Athanasopoulos, George. "Forecasting: Principles and Practice"
"""
const US_CHANGE = joinpath(dirname(@__DIR__()), "datasets", "uschange.csv")

@doc raw"""
    SUNSPOTS_YEAR

Yearly numbers of sunspots from 1700 to 1988 (rounded to one digit).
Federal Reserve Bank of St Louis.

# References
 * H. Tong (1996) Non-Linear Time Series. Clarendon Press, Oxford, p. 471.
"""
const SUNSPOTS_YEAR = joinpath(dirname(@__DIR__()), "datasets", "sunspot_year.csv")
