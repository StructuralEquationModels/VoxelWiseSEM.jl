## Tutorial

This tutorial walks through a complete voxel-wise SEM analysis.  
Dataset: Midnight Scan Club (MSC). 
OpenNeuro: ```https://openneuro.org/datasets/ds000224/versions/1.0.4```

## Dataset 

The MSC dataset contains 10 subjects (sub-MSC01 … sub-MSC10). Each subject has:
- 2 structural sessions: `ses-struct01`, `ses-struct02` → T1w and T2w images
- 10 functional sessions: `ses-func01` ... `ses-func10` 

In this tutorial we work with the T1w structural images.
Each subject contributes 2 T1w scans acquired weeks apart, giving us a small longitudinal dataset.  

We model this with a SEM (latent intercept) fitted independently at every voxel inside a brain mask.

## Download the data

You have two options for setting up the Midnight Scan Club (MSC) data for this tutorial:

### Option A: Using Julia Artifacts (Recommended & Automatic)
The dataset is distributed as a platform-independent Julia Artifact. The first time you run the script, Julia will automatically download and cache the 10-subject (with 2 sessions each) of the Midnight Scan Club dataset. 

If using this option, you do not need to install the AWS CLI or download any files manually.

### Option B: Manual Download via AWS CLI (Alternative)
If you want to use more subjects (up to the full 10 subjects) or download the data manually, you can use the AWS CLI.

**Prerequisites:**
Before running this option, install the AWS CLI on your system:
- **Mac:** `brew install awscli`
- **Linux:** install using package manager or see official guide.
- **Windows:** Download and run the installer from https://aws.amazon.com/cli/

Verify AWS CLI:
```bash
aws --version
```

Once AWS CLI is installed, run this from your terminal to download the 10 subjects:
```bash
aws s3 sync \
    --no-sign-request \
    s3://openneuro.org/ds000224 \
    ./data/ds000224 \
    --exclude "*" \
    --include "sub-MSC01/ses-struct*/anat/*T1w*" \
    --include "sub-MSC02/ses-struct*/anat/*T1w*" \
    --include "sub-MSC03/ses-struct*/anat/*T1w*" \
    --include "sub-MSC04/ses-struct*/anat/*T1w*" \
    --include "sub-MSC05/ses-struct*/anat/*T1w*" \
    --include "sub-MSC06/ses-struct*/anat/*T1w*" \
    --include "sub-MSC07/ses-struct*/anat/*T1w*" \
    --include "sub-MSC08/ses-struct*/anat/*T1w*" \
    --include "sub-MSC09/ses-struct*/anat/*T1w*" \
    --include "sub-MSC10/ses-struct*/anat/*T1w*" \
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
using LazyArtifacts

# --- Choose Data Source Option ---
# Option A (Recommended): Use automated Julia Artifacts (10 subjects)
dataset_dir = artifact"msc_dataset"

# Option B: Use manual AWS sync data (10 subjects)
# dataset_dir = "data/ds000224"
```

# The brain mask is loaded directly as a platform-independent Julia Artifact
```julia
mask_path   = joinpath(artifact"brain_mask", "brain_mask.nii.gz")

mkpath("data/measurements")
mkpath("data/results")
```

## Step 1 — Measurements

`generate_measurements` scans the BIDS directory and returns a DataFrame with one row per NIfTI file. 
The `modality` argument tells it which subfolder to look in. It automatically keeps only `.nii` and `.nii.gz` files.

The resulting DataFrame has columns:
- `subject`        e.g. "sub-MSC01"
- `subject_number` integer index (1-based, alphabetical order)  
- `session`        e.g. "ses-struct01"
- `session_number` integer parsed from the trailing digits of the session label  
- `modality`       "anat"  
- `file`           relative path to scan, e.g. "sub-MSC01/ses-struct01/anat/sub-MSC01_ses-struct01_run-01_T1w.nii.gz"  

The anat/ folder contains both T1w and T2w images. We keep only T1w scans for this tutorial.  

```julia
measurements = generate_measurements(dir = dataset_dir, modality = "anat")
filter!(r -> endswith(r.file, "_T1w.nii.gz"), measurements)
save_measurements(measurements, "data/measurements/measurements.csv")
```

Expected output:

```
 Row │ subject    subject_number  session       session_number  modality  file
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ sub-MSC01               1  ses-struct01               1  anat      sub-MSC01/ses-struct01/anat/sub-…
   2 │ sub-MSC01               1  ses-struct01               1  anat      sub-MSC01/ses-struct01/anat/sub-…
   3 │ sub-MSC01               1  ses-struct02               2  anat      sub-MSC01/ses-struct02/anat/sub-…
   4 │ sub-MSC01               1  ses-struct02               2  anat      sub-MSC01/ses-struct02/anat/sub-…
```

## Step 2a — Brain Mask Selection

A brain mask is a binary NIfTI volume (`1` = inside brain, `0` = outside) that tells `generate_coordinates` which voxels to include in the analysis.

In this tutorial, we load the brain mask directly using the Julia artifact system, which retrieves a pre-defined 11×11×11 voxel mask in the center of the brain volume.

```julia
mask_path = joinpath(artifact"brain_mask", "brain_mask.nii.gz")
```

## Step 2b — Generate voxel coordinates from the mask

`generate_coordinates` reads the mask and returns a DataFrame with one row per in-mask voxel containing its linear index (`voxel`) and 3D coordinates (`x`, `y`, `z`).

```julia
coordinates = generate_coordinates(mask = mask_path)
println("voxels in mask: ", nrow(coordinates))
```

## Step 2c — Reshape data into a 3D array

`voxel_wise_data` loads every NIfTI file and assembles a 3D array of shape `(n_voxels × n_sessions × n_subjects)`.

- Axis 1 is addressed by `coordinates.voxel_idx` (linear index in the full volume).
- Axis 2 is addressed by `measurements.session_number` (1, 2, …).
- Axis 3 is addressed by `measurements.subject_number` (1, 2, …).

For this tutorial the array shape will be (max_voxel_index, 2, 10), where 2 sessions and 10 subjects each contribute one T1w scan.

```julia
vw_data = voxel_wise_data(dataset_dir, measurements, coordinates)
println("data array size: ", size(vw_data))
save_voxel_wise_data(vw_data, "data/vw_data.jld2")
```

Indexing examples:  
- `vw_data[coordinates.voxel_idx[1], :, :]` — one voxel, all sessions × subjects.
- `vw_data[:, 1, 1]`                         — all voxels, session 1, subject 1

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
```

───────────────────3c. Outlier removal using MAD (Median Absolute Deviation)───────────────────

```julia
coordinates = step_mad!(coordinates, log, vw_data; mad_cutoff = 2.5)
```

───────────────────3d. Remove voxels with too many outliers───────────────────

```julia
coordinates = step_rm_voxel!(coordinates, log, vw_data; voxel_cutoff = 0.2)

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

Define a one-factor SEM with one latent variable `I` (intercept) and one observed variable per subject.

```julia
n_subjects    = maximum(measurements.subject_number)
observed_vars = Symbol.(:t, 1:n_subjects)
latent_vars   = [:I]

graph = @StenoGraph begin
    I → [fixed(1)] .* _(observed_vars)

    # variances
    I ↔ label(:var_I) * I                          # latent variance
    _(observed_vars) ↔ [label(:error)] .* _(observed_vars)  # residual variance

    # mean structure
    Symbol("1") → label(:mean_I) * I               # latent mean
    Symbol("1") → [fixed(0)] .* _(observed_vars)     # observed intercepts fixed to 0
end

partable = ParameterTable(
    graph;
    observed_vars = observed_vars,
    latent_vars   = latent_vars
)

base_data = vw_data[coordinates.voxel_idx[1], :, :]   # shape: (n_sessions × n_subjects)

model = Sem(
    specification = partable,
    observed      = SemObservedMissing,
    loss          = SemFIML,
    meanstructure = true,
    data          = base_data,
    implied       = RAMSymbolic
)

function fit_to_voxel(voxel_matrix; model, specification)
    # The matrix is in the expected shape (n_sessions × n_subjects)
    model_vox = replace_observed(
        model; 
        data = voxel_matrix, specification
    )
    fitted = fit(model_vox; start_val = start_simple)

    # collect parameter names and estimates into a NamedTuple
    # apply_voxelwise concatenates these into columns of the results DataFrame
    out = param_labels(fitted) .=> solution(fitted)
    push!(out, :converged => convergence(fitted))
    return NamedTuple(out)
end

results = apply_voxelwise(fit_to_voxel, coordinates, vw_data;
    model = model, specification = partable)

CSV.write("data/results/voxel_wise_results.csv", results)
```
