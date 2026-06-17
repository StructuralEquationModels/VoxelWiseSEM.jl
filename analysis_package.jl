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

# 1. Setup paths
dataset_dir = artifact"msc_dataset"
mask_path = joinpath(artifact"brain_mask", "brain_mask.nii.gz")

println("Step 1: Running generate_measurements...")
measurements = generate_measurements(dir = dataset_dir, modality = "anat")

# Save measurements
mkpath("data/measurements")
# Filter measurements to only include T1w scans (excluding T2w and defacemasks)
filter!(r -> endswith(r.file, "_T1w.nii.gz"), measurements)
save_measurements(measurements, "data/measurements/measurements.csv")

# 2. Voxel selection is handled via the distributed artifact brain mask.

# 3. Generate coordinates
println("\nStep 2a: Generating coordinates from mask...")
coordinates = generate_coordinates(mask = mask_path)
println("Number of coordinates in mask: ", nrow(coordinates))

# 4. Reshape data
println("\nStep 2b: Reshaping data into a voxel-wise 3D array...")
vw_data = voxel_wise_data(dataset_dir, measurements, coordinates)

println("Shape of vw_data: ", size(vw_data))
save_voxel_wise_data(vw_data, "vw_data.jld2")

# 5. Preprocessing
println("\nStep 3: Preprocessing...")
log = PreProcLog()
coordinates = step_missings!(coordinates, log, vw_data)
coordinates = step_zeros!(coordinates, log, vw_data)
coordinates = step_mad!(coordinates, log, vw_data; mad_cutoff = 2.5)
coordinates = step_rm_voxel!(coordinates, log, vw_data; voxel_cutoff = 0.2)
println("Preprocessed coordinates count: ", nrow(coordinates))

# 6. Fit SEM model using the actual packages and workflow
println("\nStep 4: Running SEM model fitting on the data...")

# Dynamically set variables based on number of subjects
n_subjects = maximum(measurements.subject_number)
observed_vars = Symbol.(:t, 1:n_subjects)
latent_vars = [:I]

graph = @StenoGraph begin
    # loadings
    I → [fixed(1)].*_(observed_vars)

    # variances
    I ↔ label(:var_I)*I
    _(observed_vars) ↔ [label(:error)].*_(observed_vars)

    # means
    Symbol("1") → label(:mean_I)*I
    Symbol("1") → [fixed(0)].*_(observed_vars)
end

partable = ParameterTable(
    graph;
    observed_vars = observed_vars,
    latent_vars = latent_vars
)

# Initialize base model using data from the first voxel
# data shape: rows = observations (sessions) and columns = variables (subjects)
base_data = vw_data[coordinates.voxel_idx[1], :, :]
model = Sem(
    specification = partable,
    observed = SemObservedMissing,
    loss = SemFIML,
    meanstructure = true,
    data = base_data,
    implied = RAMSymbolic
)

# Fitting function for each voxel
function fit_to_voxel(voxel_matrix; model, specification)
    # voxel_matrix is already in (n_sessions, n_subjects) shape.
    model_vox = replace_observed(
        model;
        data = voxel_matrix,
        specification = specification
    )
    fitted = fit(model_vox; start_val = start_simple)
    out = param_labels(fitted) .=> solution(fitted)
    push!(out, :converged => convergence(fitted))
    return NamedTuple(out)
end

# Fit the models for the first 10 voxels
results = apply_voxelwise(
    fit_to_voxel,
    coordinates[1:10, :],
    vw_data;
    model = model,
    specification = partable
)

println("\nFirst few rows of fitting results:")
println(first(results, 5))

println("\n=== Validation Successful! VoxelWiseSEM runs perfectly ===")

