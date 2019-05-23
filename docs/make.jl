using Documenter, StateSpaceModels

makedocs(
    modules = [StateSpaceModels],
    doctest  = false,
    clean    = true,
    format   = Documenter.HTML(),
    sitename = "StateSpaceModels.jl",
    authors = "Raphael Saavedra, Mario Souto, Guilherme Bodin and contributors.",
    pages = [
        "Home" => "index.md",
        "manual.md"
    ]
)

deploydocs(
    repo = "github.com/LAMPSPUC/StateSpaceModels.jl.git",
)