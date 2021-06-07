using Documenter
using StateSpaceModels

# Set up to run docstrings with jldoctest
DocMeta.setdocmeta!(
    StateSpaceModels, :DocTestSetup, :(using StateSpaceModels); recursive=true
)

makedocs(;
    modules=[StateSpaceModels],
    doctest=true,
    clean=true,
    format=Documenter.HTML(mathengine=Documenter.MathJax2()),
    sitename="StateSpaceModels.jl",
    authors="Raphael Saavedra, Guilherme Bodin, and Mario Souto",
    pages=[
        "Home" => "index.md",
        "manual.md",
        "examples.md"
    ],
)

deploydocs(
        repo="github.com/LAMPSPUC/StateSpaceModels.jl.git",
        push_preview = true
    )
