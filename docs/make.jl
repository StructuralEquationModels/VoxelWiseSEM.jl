using Documenter, VoxelWiseSEM

makedocs(
    sitename = "VoxelWiseSEM.jl",
    
    pages = [
        "Home"     => "index.md",
        "Tutorial" => "tutorial.md",
        "API"      => "api.md",
    ],
    modules = [VoxelWiseSEM],
    format   = Documenter.HTML(
        prettyurls = true,
    ),
    warnonly = true,        
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/StructuralEquationModels/VoxelWiseSEM.jl",
    devbranch = "main",
)