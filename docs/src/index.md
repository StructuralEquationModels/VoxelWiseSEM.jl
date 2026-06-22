# VoxelWiseSEM.jl 🧠

A Julia package for voxel-wise Structural Equation Modeling on brain MRI data.

Instead of analysing the whole brain at once, VoxelWiseSEM fits a structural 
equation model independently at every voxel inside a brain mask, then collects 
all results into a single DataFrame that can be mapped back to a brain image 
for visualisation.

## Get Started →
To get started, we recommend the following order:

1. install the package [Installation],
2. read [Tutorial](@ref), and
3. get familiar with Our Concept of a Structural Equation Model.

See the [Tutorial](@ref) for a full working example using the Midnight Scan Club dataset.


## Installation

You must have julia installed (and we strongly recommend using an IDE of your choice; we like VS Code with the Julia extension).

To install the latest version of our package, use the following commands:

```julia
using Pkg
Pkg.add(url = "https://github.com/StructuralEquationModels/VoxelWiseSEM.jl")
```

## Pipeline overview

The package follows a four-step pipeline:

**Step 1 — Check available measurements**<br>
Scan the BIDS dataset folder and collect all scan files into a table.

**Step 2 — Reshape data**<br>
Generate voxel coordinates from a brain mask. Load every NIfTI volume
and reshape into a single 3D array of shape `(voxels × subjects × sessions)`.

**Step 3 — Preprocess and log**<br>
Remove bad voxels — entirely missing, entirely zero, or too noisy.
Every step is recorded in a `PreProcLog` for full reproducibility.

**Step 4 — Fit models**<br>
Apply any SEM (or any function) independently at every voxel.
Results are collected into a DataFrame with one row per voxel,
ready to map back to a brain image for visualisation.
