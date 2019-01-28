using Documenter, StateSpaceModels

makedocs(
    modules = [StateSpaceModels],
    doctest  = false,
    clean    = true,
    format   = :html,
    sitename = "StateSpaceModels.jl",
    authors = "Raphael Saavedra, Mario Souto and contributors.",
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "manual.md"
            ]
    ]
)

deploydocs(
    repo = "github.com/LAMPSPUC/StateSpaceModels.jl.git",
)