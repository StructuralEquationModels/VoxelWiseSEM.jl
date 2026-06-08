## Tutorial

This tutorial walks through a complete voxel-wise SEM analysis. 
Dataset: Midnight Scan Club (MSC). 
OpenNeuro: ```https://openneuro.org/datasets/ds000224/versions/1.0.4```

## Dataset 

The MSC dataset contains 10 subjects (sub-MSC01 … sub-MSC10). Each subject has:
- 2 structural sessions: 'ses-struct01', `ses-struct02` → T1w and T2w images
- 10 functional sessions: 'ses-func01' ...`ses-func10` 

In this tutorial we work with the T1w structural images.
Each subject contributes 2 T1w scans acquired weeks apart, giving us a small longitudinal dataset.  

We model this with a SEM (latent intercept) fitted independently at every voxel inside a brain mask.

## Download the data

Download the dataset (2 subjects is enough to follow this tutorial):

## Prerequisites

Before running this tutorial, install the AWS CLI on your system:

**Mac:**
```bash
brew install awscli
```

**Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**Windows:**
Download and run the installer from https://aws.amazon.com/cli/

Verify it worked:
```bash
aws --version
```

Once AWS CLI is installed, run this from your terminal:

```bash
aws s3 sync \
    --no-sign-request \
    s3://openneuro.org/ds000224 \
    ./data/ds000224 \
    --exclude "*" \
    --include "sub-MSC01/ses-struct*/anat/*T1w*" \
    --include "sub-MSC02/ses-struct*/anat/*T1w*" \
    --include "participants.tsv" \
    --include "dataset_description.json"
```

Then start Julia and follow the steps below.

## Setup

```julia
using Pkg

Pkg.activate(".")

using VoxelWiseSEM
using DataFrames, NIfTI, CSV, JLD2, Statistics
using StenoGraphs, StructuralEquationModels

dataset_dir = "data/ds000224"
mask_path   = "data/brain_mask.nii.gz"
mkpath("data/measurements")
mkpath("data/results")
```

## Step 1 — Measurements

`generate_measurements` scans the BIDS directory and returns a DataFrame with one row per NIfTI file. 
The `modality` argument tells it which subfolder to look in. It automatically keeps only `.nii` and `.nii.gz` files.

The resulting DataFrame has columns:

- subject        e.g. "sub-MSC01"
- subject_number integer index (1-based, alphabetical order)  
- session        e.g. "ses-struct01"
- session_number integer parsed from the trailing digits of the session label  
- modality       "anat"  
- file           e.g. "sub-MSC01_ses-struct01_T1w.nii.gz"  

The anat/ folder contains both T1w and T2w images. We keep only T1w scans for this tutorial.  

```julia
measurements = generate_measurements(dir = dataset_dir, modality = "anat")
filter!(r -> endswith(r.file, "_T1w.nii.gz"), measurements)
save_measurements(measurements, "data/measurements/measurements.csv")
```

Expected output:

```
 Row │ subject    subject_number  session       session_number  modality  file
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ sub-MSC01  1               ses-struct01  1               anat      sub-MSC01_ses-struct01_T1w.nii.gz
   2 │ sub-MSC01  1               ses-struct02  2               anat      sub-MSC01_ses-struct02_T1w.nii.gz
   3 │ sub-MSC02  2               ses-struct01  1               anat      sub-MSC02_ses-struct01_T1w.nii.gz
   4 │ sub-MSC02  2               ses-struct02  2               anat      sub-MSC02_ses-struct02_T1w.nii.gz
```

## Step 2a — Create a brain mask

A brain mask is a binary NIfTI volume (`1` = inside brain, `0` = outside). It tells generate_coordinates
which voxels to include in the analysis.  

For this tutorial we create a small 5×5×5 voxel mask in the centre of the
volume so the pipeline runs quickly on any machine.

```julia
ref_path = joinpath(dataset_dir,
    measurements[1, :subject], measurements[1, :session],
    "anat", measurements[1, :file])
img = niread(ref_path)

img.raw .= 0.0f0
x_mid, y_mid, z_mid = size(img) .÷ 2
img.raw[x_mid-2:x_mid+2, y_mid-2:y_mid+2, z_mid-2:z_mid+2] .= 1.0f0

niwrite(mask_path, img)
```

## Step 2b — Generate voxel coordinates from the mask

`generate_coordinates` reads the mask and returns a DataFrame with one row
per in-mask voxel containing its linear index (`voxel`) and 3D coordinates
(`x`, `y`, `z`).

```julia
coordinates = generate_coordinates(mask = mask_path)
println("voxels in mask: ", nrow(coordinates))  # → 125
```

## Step 2c — Reshape data into a 3D array

`voxel_wise_data` loads every NIfTI file and assembles a 3D array of shape
`(n_voxels × n_subjects × n_sessions)`.

Axis 1 is addressed by coordinates.voxel (linear index in the full volume).

Axis 2 is addressed by measurements.subject_number  (1, 2, …).

Axis 3 is addressed by measurements.session_number  (1, 2, …).

For this tutorial the array shape will be (max_voxel_index, 2, 2), where 2 subjects and 2 sessions each contribute one T1w scan.

```julia
vw_data = voxel_wise_data(dataset_dir, measurements, coordinates)
println("data array size: ", size(vw_data))  # → (max_voxel_index, 2, 2)
save_voxel_wise_data(vw_data, "data/vw_data.jld2")
```

Indexing examples:  
    vw_data[coordinates.voxel[1], :, :]  — one voxel, all subjects × sessions (2×2 matrix). 
    vw_data[:, 1, 1]                     — all voxels, subject 1, session 1  

## Step 3 — Preprocessing

```julia
log = PreProcLog()
```

───────────────────  3a. Remove voxels that are entirely missing ────────────────────

Handles voxels that fall inside the mask but have no valid data for any subject/session  
Adds column `"n_nonmissing"` to coordinates. Voxels where `n_nonmissing` == 0 are removed.

```julia
coordinates = step_missings!(coordinates, log, vw_data)
```

─────────────────── 3b. Remove voxels that are entirely zero ───────────────────

Adds column `"p_zero"` (proportion of zero values). Voxels where `p_zero`== 1 (all zeros) are removed.  

```julia
coordinates = step_zeros!(coordinates, log, vw_data)
println("voxels remaining after preprocessing: ", nrow(coordinates))
```

Inspect the log to see what was removed at each step. Save the log.

```julia
println(log)
save_log(log, (modality = "T1w",))
```
example output:  
Step 1:  
    step_missings!  
    Dict("all_missing" => 0)  
Step 2:  
    step_zeros!  
    Dict("all_zero" => 0, "some_zero" => 0)  


## Step 4 — Fit SEM at every voxel

Define a one-factor SEM with one latent variable `I` (intercept) and one
observed variable per subject.

```julia
n_subjects    = maximum(measurements.subject_number)
observed_vars = Symbol.(:t, 1:n_subjects)
latent_vars   = [:I]

graph = @StenoGraph begin
    I → fixed(1) .* _(observed_vars)
    I ↔ label(:var_I) * I
    _(observed_vars) ↔ label(:error) .* _(observed_vars)
    Symbol("1") → label(:mean_I) * I
    Symbol("1") → fixed(0) .* _(observed_vars)
end

partable = ParameterTable(graph;
    observed_vars = observed_vars,
    latent_vars   = latent_vars)

base_data = vw_data[coordinates.voxel[1], :, :]'
model = Sem(
    specification = partable,
    observed      = SemObservedMissing,
    loss          = SemFIML,
    meanstructure = true,
    data          = base_data,
    implied       = RAMSymbolic)

function fit_to_voxel(voxel_matrix; model, specification)
    model_vox = replace_observed(model;
        data = voxel_matrix', specification = specification)
    fitted = fit(model_vox; start_val = start_simple)
    out = param_labels(fitted) .=> solution(fitted)
    push!(out, :converged => convergence(fitted))
    return NamedTuple(out)
end

results = apply_voxelwise(fit_to_voxel, coordinates, vw_data;
    model = model, specification = partable)

CSV.write("data/results/voxel_wise_results.csv", results)
```

