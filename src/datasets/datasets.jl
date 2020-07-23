# Datasets are not exported to avoid colliding with names in the Main scope

@doc raw"""
    NILE

The absolute path for the so called `NILE` dataset stored inside StateSpaceModels.jl.
This dataset provides measurements of the annual flow of the Nile river at Aswan from 1871 to 1970, 
in ``10^8 m^3``.

See more on [Nile river annual flow](@ref)

# References
 * Durbin, James, & Siem Jan Koopman. (2012). "Time Series Analysis by State Space Methods: Second Edition." Oxford University Press. (Chapter 2)
"""
const NILE = joinpath(@__DIR__(), "nile.csv")

@doc raw"""
The absolute path for the so called `AIRPASSENGERS` dataset stored inside StateSpaceModels.jl.
This dataset provides monthly totals of a US airline passengers from 1949 to 1960.

See more on [Log of airline passengers](@ref)

# References 
 * https://www.stata-press.com/data/r12/ts.html
"""
const AIRPASSENGERS = joinpath(@__DIR__(), "airpassengers.csv")