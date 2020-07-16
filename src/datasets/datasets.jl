# Datasets are not exported to avoid colliding with names in the Main scope

"""
"""
const NILE = Float64.(readdlm(joinpath(@__DIR__(), "Nile.csv"), ',')[2:end, 2])

"""
"""
const AIRPASSENGERS = Float64.(readdlm(joinpath(@__DIR__(), "AirPassengers.csv"), ',')[2:end, 2])