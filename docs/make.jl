using Documenter, StateSpaceModels

makedocs(
    modules = [StateSpaceModels],
    doctest  = false,
    clean    = true,
    format   = Documenter.HTML(),
    sitename = "StateSpaceModels.jl",
    authors = "Raphael Saavedra, Guilherme Bodin, and Mario Souto",
    pages = [
        "Home" => "index.md",
        "manual.md",
        "examples.md",
        "reference.md"
    ]
)

deploydocs(
    repo = "github.com/LAMPSPUC/StateSpaceModels.jl.git",
)