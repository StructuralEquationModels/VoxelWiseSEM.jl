module VoxelWiseSEM
    using DataFrames, NIfTI, CSV, ProgressMeter, JLD2
        include("generate_measurements.jl")
        include("generate_coordinates.jl")
        include("voxel_wise_data.jl")
        include("apply_voxelwise.jl")
    export 
        generate_measurements,
        generate_coordinates,
        voxel_wise_data,
        save_voxel_wise_data,
        apply_voxelwise,
        modify_voxelwise!
end
