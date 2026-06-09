############################################################################################
#
#  VoxelWiseSEM.jl — Tutorial
#
#  Dataset: Midnight Scan Club (MSC)
#  OpenNeuro: https://openneuro.org/datasets/ds000224/versions/1.0.4
#
#  The MSC dataset contains 10 subjects (sub-MSC01 … sub-MSC10).
#  Each subject has:
#    - 2 structural sessions: ses-struct01, ses-struct02  →  T1w and T2w images
#    - 10 functional sessions: ses-func01 … ses-func10   →  resting-state fMRI
#
#  In this tutorial we work with the T1w structural images.
#  Each subject contributes 2 T1w scans acquired weeks apart, giving us a
#  small longitudinal dataset. The scientific question is:
#
#       "Is there a systematic difference in grey-matter intensity
#        between session 1 and session 2 across subjects?"
#
#  We model this with a one-factor SEM (latent intercept) fitted
#  independently at every voxel inside a brain mask.
#
############################################################################################
#
#  Download the dataset (2 subjects is enough to follow this tutorial):
#
#    pip install awscli
#
#    aws s3 sync \
#    --no-sign-request \
#    s3://openneuro.org/ds000224 \
#    ./data/ds000224 \
#    --exclude "*" \
#    --include "sub-MSC01/ses-struct*/anat/*T1w*" \
#    --include "sub-MSC01/ses-struct*/anat/*T2w*" \
#    --include "sub-MSC02/ses-struct*/anat/*T1w*" \
#    --include "sub-MSC02/ses-struct*/anat/*T2w*" \
#    --include "participants.tsv" \
#    --include "dataset_description.json"

#  After the download your folder should look like:
#
#    data/ds000224/
#      sub-MSC01/
#        ses-struct01/anat/sub-MSC01_ses-struct01_T1w.nii.gz
#        ses-struct02/anat/sub-MSC01_ses-struct02_T1w.nii.gz
#      sub-MSC02/
#        ses-struct01/anat/sub-MSC02_ses-struct01_T1w.nii.gz
#        ses-struct02/anat/sub-MSC02_ses-struct02_T1w.nii.gz
#
############################################################################################

using Pkg
Pkg.activate(".")

using VoxelWiseSEM
using DataFrames
using NIfTI
using CSV
using JLD2
using Statistics
using StenoGraphs
using StructuralEquationModels

###########################################################################################
#  Paths

dataset_dir = "data/ds000224"
mask_path   = "data/brain_mask.nii.gz" # we will create this mask further down
mkpath("data/measurements")
mkpath("data/results")

###########################################################################################
# STEP 1 — Check available measurements and create measurements.csv

# generate_measurements scans the BIDS directory and returns a DataFrame
# with one row per NIfTI file found inside the chosen modality folder.

# modality = "anat" tells the function to look inside each session's anat/
# subfolder. It automatically finds only .nii and .nii.gz files, so JSON
# sidecar files are already excluded.

# The resulting DataFrame has columns:
#   subject        e.g. "sub-MSC01"
#   subject_number integer index (1-based, alphabetical order)
#   session        e.g. "ses-struct01"
#   session_number integer parsed from the trailing digits of the session label
#   modality       "anat" 
#   file           e.g. "sub-MSC01_ses-struct01_T1w.nii.gz"

measurements = generate_measurements(dir = dataset_dir, modality = "anat")

# The anat/ folder contains both T1w and T2w images.
# We keep only T1w scans for this tutorial.
filter!(r -> endswith(r.file, "_T1w.nii.gz"), measurements)

# Inspect the table 
println(first(measurements, 4))
# expected:
#  Row │ subject    subject_number  session       session_number  modality  file
#      │ String     Int64           String        Int64           String    String
# ─────┼──────────────────────────────────────────────────────────────────────────────
#    1 │ sub-MSC01  1               ses-struct01  1               anat      sub-MSC01_ses-struct01_T1w.nii.gz
#    2 │ sub-MSC01  1               ses-struct02  2               anat      sub-MSC01_ses-struct02_T1w.nii.gz
#    3 │ sub-MSC02  2               ses-struct01  1               anat      sub-MSC02_ses-struct01_T1w.nii.gz
#    4 │ sub-MSC02  2               ses-struct02  2               anat      sub-MSC02_ses-struct02_T1w.nii.gz

# Save to CSV.
save_measurements(measurements, "data/measurements/measurements.csv")

############################################################################################
# STEP 2a — Create a brain mask
############################################################################################

# A brain mask is a binary NIfTI volume (1 = inside brain, 0 = outside) in the
# same space and dimensions as the T1w images. It tells generate_coordinates
# which voxels to include in the analysis.

# For this tutorial we create a small
# 5×5×5 voxel mask in the centre of the volume so the pipeline runs quickly on any machine.

# Load the first T1w image to get the volume dimensions and NIfTI header.
ref_path = joinpath(
    dataset_dir,
    measurements[1, :subject],
    measurements[1, :session],
    "anat",
    measurements[1, :file]
)
img = niread(ref_path)

# Set the entire volume to 0, then turn on a 5×5×5 block in the centre.
img.raw .= 0
x_mid, y_mid, z_mid = size(img) .÷ 2
img.raw[x_mid-2:x_mid+2, y_mid-2:y_mid+2, z_mid-2:z_mid+2] .= 1.0f0

niwrite(mask_path, img)
println("mask written to: ", mask_path, "  (", sum(img.raw .== 1), " voxels)")

############################################################################################
# STEP 2b — Generate voxel coordinates from the mask
############################################################################################

# generate_coordinates reads the mask and returns a DataFrame with one row per
# in-mask voxel. 

coordinates = generate_coordinates(mask = mask_path)
############################################################################################
# STEP 2c — Reshape BIDS volumes into a voxel-wise 3D array
############################################################################################

# voxel_wise_data loads every NIfTI file listed in measurements and assembles
# a single 3D array of shape:
#
#   (n_voxels  ×  n_subjects  ×  n_sessions)
#    axis 1       axis 2         axis 3
#
# Axis 1 is addressed by coordinates.voxel (linear index in the full volume).
# Axis 2 is addressed by measurements.subject_number  (1, 2, …).
# Axis 3 is addressed by measurements.session_number  (1, 2, …).
#
# For this tutorial the array shape will be (max_voxel_index, 2, 2)
# where 2 subjects and 2 sessions each contribute one T1w scan.

vw_data = voxel_wise_data(dataset_dir, measurements, coordinates)

println("data array size: ", size(vw_data))
# → (max_voxel_index, 2, 2)

# Indexing examples:
#   vw_data[coordinates.voxel[1], :, :]  — one voxel, all subjects × sessions (2×2 matrix)
#   vw_data[:, 1, 1]                     — all voxels, subject 1, session 1

# Save to JLD2
save_voxel_wise_data(vw_data, "data/vw_data.jld2")

############################################################################################
# STEP 3 — Preprocessing and logging
############################################################################################

# PreProcLog records 
log = PreProcLog()

# ── 3a. Remove voxels that are entirely missing ───────────────────────────────

# Handles voxels that fall inside the mask but have no valid data for any
# subject/session
# Adds column "n_nonmissing" to coordinates. Voxels where n_nonmissing == 0
# are removed.

coordinates = step_missings!(coordinates, log, vw_data)

# ── 3b. Remove voxels that are entirely zero ──────────────────────────────────

# Adds column "p_zero" (proportion of zero values). Voxels where p_zero == 1
# (all zeros) are removed.

coordinates = step_zeros!(coordinates, log, vw_data)

println("voxels remaining after preprocessing: ", nrow(coordinates))

# Inspect the log to see what was removed at each step.
println(log)
# example output:
#   Step 1:
#       step_missings!
#       Dict("all_missing" => 0)
#   Step 2:
#       step_zeros!
#       Dict("all_zero" => 0, "some_zero" => 0)

# Save the log. condition_filename turns the named tuple into a filename string,
# e.g. (modality="T1w",) → "modality_T1w.jld2"
save_log(log, (modality = "T1w",))


############################################################################################
#  STEP 4 — Fit a SEM model at every voxel
############################################################################################

# We fit a one-factor SEM at each voxel. The model has:
#   - one latent variable I 
#   - one observed variable per subject

n_subjects    = maximum(measurements.subject_number)
observed_vars = Symbol.(:t, 1:n_subjects) 
latent_vars   = [:I]

# ── 4a. Define the SEM graph with StenoGraphs ────────────────────────────────

graph = @StenoGraph begin
    I → fixed(1) .* _(observed_vars)

    # variances
    I ↔ label(:var_I) * I                          # latent variance
    _(observed_vars) ↔ label(:error) .* _(observed_vars)  # residual variance

    # mean structure
    Symbol("1") → label(:mean_I) * I               # latent mean
    Symbol("1") → fixed(0) .* _(observed_vars)     # observed intercepts fixed to 0
end

partable = ParameterTable(
    graph;
    observed_vars = observed_vars,
    latent_vars   = latent_vars
)

# ── 4b. Build the base model on the first voxel ──────────────────────────────
# vw_data[v, :, :] is (n_subjects × n_sessions).
# The SEM expects rows = observations (sessions) and columns = variables (subjects),
# so we transpose with '.

base_data = vw_data[coordinates.voxel[1], :, :]'   # shape: (n_sessions × n_subjects)

model = Sem(
    specification = partable,
    observed      = SemObservedMissing,
    loss          = SemFIML,
    meanstructure = true,
    data          = base_data,
    implied       = RAMSymbolic
)

# ── 4c. Define the per-voxel fitting function ─────────────────────────────────
# apply_voxelwise calls this function once per voxel, passing a view of the
# data array of shape (n_subjects × n_sessions).

function fit_to_voxel(voxel_matrix; model, specification)
    # transpose to (n_sessions × n_subjects) as expected by the SEM
    model_vox = replace_observed(
        model, voxel_matrix'
    )
    fitted = fit(model_vox; start_val = start_simple)

    # collect parameter names and estimates into a NamedTuple
    # apply_voxelwise concatenates these into columns of the results DataFrame
    out = param_labels(fitted) .=> solution(fitted)
    push!(out, :converged => converged(fitted))
    return NamedTuple(out)
end

# ── 4d. Run apply_voxelwise ───────────────────────────────────────────────────
# apply_voxelwise iterates over every row of coordinates, extracts the
# (n_subjects × n_sessions) data slice for that voxel using coordinates.voxel
# as the axis-1 index, calls fit_to_voxel, and concatenates the returned
# NamedTuples into a DataFrame alongside the coordinates.

results = apply_voxelwise(
    fit_to_voxel,
    coordinates,
    vw_data;
    model         = model,
    specification = partable
)

# The results DataFrame has all coordinate columns (voxel, x, y, z) plus one
# column per estimated parameter (var_I, error, mean_I, converged, …).

println("\nresults (first 5 rows):")
println(first(results, 5))

CSV.write("data/results/voxel_wise_results.csv", results)
println("results saved to data/results/voxel_wise_results.csv")


