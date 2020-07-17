# Datasets are not exported to avoid colliding with names in the Main scope

"""
    NILE

The absolute path for the Nile River dataset stored inside StateSpaceModels.jl
"""
const NILE = joinpath(@__DIR__(), "Nile.csv")

"""
"""
const AIRPASSENGERS = joinpath(@__DIR__(), "AirPassengers.csv")