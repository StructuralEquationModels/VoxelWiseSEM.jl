module VoxelWiseSEM
    using StatsBase, DataFrames, NIfTI, CSV, ProgressMeter, JLD2
        include("generate_measurements.jl")
        include("generate_coordinates.jl")
        include("voxel_wise_data.jl")
        include("apply_voxelwise.jl")
        include("preproc.jl")
        include("helper.jl")
        include("logs.jl")
    export 
        generate_measurements,
        generate_coordinates,
        voxel_wise_data,
        save_voxel_wise_data,
        apply_voxelwise,
        modify_voxelwise!,
        # preprocessing
        step_missings!,
        step_zeros!,
        n_nonmissing,
        p_zero,
        all_zero,
        compute_mad,
        set_missing_mad!,
        step_mad!,
        step_rm_voxel!,
        # cluster
        slurm_array_id,
        condition_filename,
        save_log,
        # logs
        PreProcLog
end
