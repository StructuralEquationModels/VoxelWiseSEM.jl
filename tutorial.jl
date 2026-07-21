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
#  small longitudinal dataset. 

#  We model this with a one-factor SEM (latent intercept) fitted
#  independently at every voxel inside a brain mask.
#
############################################################################################
#  Dataset Download:
#  The dataset is distributed as a platform-independent Julia Artifact.
#  The first time you run this tutorial, Julia will automatically download and cache
#  the 10 subjects of the Midnight Scan Club dataset
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
using LazyArtifacts

# Fallback names method for NamedTuple to support save_log in the tutorial environment
Base.names(nt::NamedTuple) = collect(keys(nt))

###########################################################################################
#  Setup Paths

dataset_dir = artifact"msc_dataset"
mask_path   = joinpath(artifact"brain_mask", "brain_mask.nii.gz")
mkpath("data/measurements")
mkpath("data/results")
mkpath("logs")

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

println("Step 1: Running generate_measurements...")
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

# 2a. Voxel selection is handled via the distributed artifact brain mask.

############################################################################################
# STEP 2b — Generate voxel coordinates from the mask
############################################################################################

# generate_coordinates reads the mask and returns a DataFrame with one row per
# in-mask voxel. 

println("\nStep 2b: Generating coordinates from mask...")
coordinates = generate_coordinates(mask = mask_path)
############################################################################################
# STEP 2c — Reshape BIDS volumes into a voxel-wise 3D array
############################################################################################

# voxel_wise_data loads every NIfTI file listed in measurements and assembles
# a single 3D array of shape:
#
#   (n_voxels  ×  n_sessions  ×  n_subjects)
#    axis 1       axis 2         axis 3
#
# Axis 1 is addressed by coordinates.voxel_idx.
# Axis 2 is addressed by measurements.session_number  (1, 2, …).
# Axis 3 is addressed by measurements.subject_number  (1, 2, …).
#
# For this tutorial the array shape will be (n_voxels, 2, 10)
# where 10 subjects and 2 sessions each contribute one T1w scan.

vw_data = voxel_wise_data(dataset_dir, measurements, coordinates)

println("data array size: ", size(vw_data))
# → (n_voxels, 2, 10)

# Indexing examples:
#   vw_data[coordinates.voxel_idx[1], :, :]  — one voxel, all sessions × subjects (2×10 matrix)
#   vw_data[:, 1, 1]                         — all voxels, session 1, subject 1

# Save to JLD2
save_voxel_wise_data(vw_data, "data/vw_data.jld2")

############################################################################################
# STEP 3 — Preprocessing and logging
############################################################################################

println("\nStep 3: Preprocessing...")
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

# ── 3c. Outlier removal using MAD (Median Absolute Deviation) ─────────────────

# Sets values more than mad_cutoff * MAD away from the median to missing.
# Voxels with a MAD of 0 are removed.

coordinates = step_mad!(coordinates, log, vw_data; mad_cutoff = 2.5)

# ── 3d. Remove voxels with too many outliers ──────────────────────────────────

# Removes voxels where more than voxel_cutoff (fraction) of data points were removed as outliers.

coordinates = step_rm_voxel!(coordinates, log, vw_data; voxel_cutoff = 0.2)

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
#   Step 3: 
#       step_mad!
#       Dict("mad_cutoff" => 2.5, "mad_zero" => 0, "removed_data_fraction" => 0.086)
#   Step 4: 
#       step_rm_voxel!
#       Dict("voxel_cutoff" => 0.2, "removed_voxel_fraction" => 0.345)

# Save the log. condition_filename turns the named tuple into a filename string,
# e.g. (modality="T1w",) → "modality_T1w.jld2"
save_log(log, (modality = "T1w",))


############################################################################################
#  STEP 4 — Fit a SEM model at every voxel
############################################################################################

# We fit a one-factor SEM at each voxel. The model has:
#   - one latent variable I 
#   - one observed variable per subject

n_sessions    = maximum(measurements.session_number)
observed_vars = Symbol.(:t, 1:n_sessions) 
latent_vars   = [:I]

# ── 4a. Define the SEM graph with StenoGraphs ────────────────────────────────

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

# ── 4b. Build the base model on the first voxel ──────────────────────────────
# vw_data[v, :, :] is (n_sessions × n_subjects).
# The SEM expects rows = observations (sessions) and columns = variables (subjects),
# which matches this layout directly.

base_data = vw_data[coordinates.voxel_idx[4], :, :]'   # shape: (n_subjects x n_sessions)

model = Sem(
    specification = partable,
    observed      = SemObservedMissing,
    loss          = SemFIML,
    meanstructure = true,
    data          = base_data,
    implied       = RAMSymbolic,
)

# ── 4c. Define the per-voxel fitting function ─────────────────────────────────
# apply_voxelwise calls this function once per voxel, passing a view of the
# data array of shape (n_sessions × n_subjects).

function fit_to_voxel(voxel_matrix; model)
    # The matrix is in the expected shape (n_sessions × n_subjects)
    model_vox = replace_observed(
        model, voxel_matrix'/100
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
# (n_sessions × n_subjects) data slice for that voxel using coordinates.voxel_idx
# as the axis-1 index, calls fit_to_voxel, and concatenates the returned
# NamedTuples into a DataFrame alongside the coordinates.

results = apply_voxelwise(
    fit_to_voxel,
    coordinates,
    vw_data;
    model         = model,
)

# The results DataFrame has all coordinate columns (voxel, x, y, z) plus one
# column per estimated parameter (var_I, error, mean_I, converged, …).

println("\nresults (first 5 rows):")
println(first(results, 5))

CSV.write("data/results/voxel_wise_results.csv", results)
println("results saved to data/results/voxel_wise_results.csv")
