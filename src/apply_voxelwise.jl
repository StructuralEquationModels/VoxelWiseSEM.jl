function apply_voxelwise(fun, coordinates, vw_data; kwargs...)
    rows = @showprogress [
        fun(vw_data[string((r.x, r.y, r.z))]; kwargs...) for r in eachrow(coordinates)
        ]
    return [coordinates DataFrame(rows)]
end

function modify_voxelwise!(fun, vw_data; kwargs...)
    @showprogress for k in keys(vw_data)
        vw_data[k] = fun(vw_data[k]; kwargs...)
    end
    return nothing
end