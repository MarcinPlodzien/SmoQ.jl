using SmoQ
using Documenter

DocMeta.setdocmeta!(SmoQ, :DocTestSetup, :(using SmoQ); recursive=true)

makedocs(;
    modules=[SmoQ],
    authors="Marcin Płodzień <95550675+MarcinPlodzien@users.noreply.github.com> and contributors",
    sitename="SmoQ.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
