############################################################################################
# apply a function to data from each voxel

function apply_voxelwise(fun, coordinates::DataFrame, data::Dict; kwargs...)
    rows = @showprogress [
        fun(data[string((r.x, r.y, r.z))]; kwargs...) for r in eachrow(coordinates)
        ]
    return [coordinates DataFrame(rows)]
end

function apply_voxelwise(fun, coordinates::DataFrame, data::Array; kwargs...)
    indices = hasproperty(coordinates, :voxel_idx) ? coordinates.voxel_idx : coordinates.voxel
    rows = @showprogress [
        fun(view(data, i, :, :); kwargs...) for i in indices
        ]
    return [coordinates DataFrame(rows)]
end