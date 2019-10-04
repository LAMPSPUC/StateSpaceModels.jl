push!(LOAD_PATH, "C:\\Users\\mdietze\\Downloads\\cmder\\StateSpaceModels.jl\\src")
using Documenter, StateSpaceModels
cd("C:\\Users\\mdietze\\Downloads\\cmder\\StateSpaceModels.jl\\docs")
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
