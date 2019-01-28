using Documenter

makedocs(
    modules = [Documenter],
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = false
    ),
    sitename = "StateSpaceModels.jl",
    authors = "Guilherme Bodin and contributors.",
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "manual.md"
            ]
    ]
)
