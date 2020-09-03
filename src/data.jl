# Datasets are not exported to avoid colliding with names in the Main scope

@doc raw"""
    NILE

The absolute path for the Nile flow series.
This series provides measurements of the annual flow of the Nile river at Aswan from 1871 to 1970, in ``10^8 m^3``.

See more in [Nile river annual flow](@ref)

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. (Chapter 2)
"""
const NILE = joinpath(dirname(@__DIR__()), "data", "nile_flow.csv")

@doc raw"""
    AIR_PASSENGERS

The absolute path for the air passengers series.
This series provides monthly totals of a US airline passengers from 1949 to 1960.

See more in [Log of airline passengers](@ref)

# References
 * https://www.stata-press.com/data/r12/ts.html
"""
const AIR_PASSENGERS = joinpath(dirname(@__DIR__()), "data", "airpassengers.csv")

@doc raw"""
    INTERNET

The absolute path for the internet series.
This series provides the number of users logged in an Internet server in each minute over 100 minutes.

See more in TODO

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. (Chapter 9)
"""
const INTERNET = joinpath(dirname(@__DIR__()), "data", "internet.csv")

@doc raw"""
    WHOLESALE_PRICE_INDEX

The absolute path for the wholesale price index series.
This series represents the quarterly US wholesale price index from 1960 to 1990.

See more in TODO

# References
 * https://www.stata.com/manuals13/tsarima.pdf
"""
const WHOLESALE_PRICE_INDEX = joinpath(dirname(@__DIR__()), "data", "wholesale_priceindex.csv")
