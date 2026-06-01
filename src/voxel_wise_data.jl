"""
    voxel_wise_data(dir, measurements, coordinates) -> Array{Union{Missing, T}, 3}
 
Reshapes BIDS brain volumes into a 3D array of voxel-wise data of size `(n_voxels, n_subjects, n_sessions)`
 
Axis 1 (`n_voxels`) is indexed by the linear voxel index from `coordinates.voxel`
Axis 2 (`n_subjects`) is indexed by `measurements.subject_number`
Axis 3 (`n_sessions`) is indexed by `measurements.session_number`
Missing subject/session combinations are filled with `missing`
 
# Arguments
- `dir`: path to the BIDS root directory
- `measurements`: DataFrame returned by `generate_measurements`
- `coordinates`: DataFrame returned by `generate_coordinates`
 
# Example
```julia
data = voxel_wise_data("/data/ds000224", measurements, coordinates)
data[:, 1, 1]    # all voxels for subject 1, session 1
data[38921, :, :] # all subjects and sessions for voxel 38921
```
"""
function voxel_wise_data(dir, measurements, coordinates)
    # first volume
    r = eachrow(measurements)[1]
    vol = read_volume(r, dir)
    number_type = eltype(vol)
    volumes = Dict((r.subject_number, r.session_number) => vol)

    println("step 1/2:")
    @showprogress for r in eachrow(measurements)[2:end]
        vol = read_volume(r, dir)
        push!(volumes, (r.subject_number, r.session_number) => vol)
    end

    n_row = maximum(measurements.subject_number)
    n_col = maximum(measurements.session_number)
    n_voxels = isempty(coordinates) ? 0 : maximum(coordinates.voxel)

    vw_data = missings(number_type, n_voxels, n_row, n_col)

    println("step 2/2:")
    @showprogress for c in eachrow(coordinates)
        v = c.voxel
        x, y, z = c.x, c.y, c.z
        for (i, j) in keys(volumes)
            vw_data[v, i, j] = volumes[(i, j)][x, y, z]
        end
    end

    return vw_data
end

"""
    save_voxel_wise_data(vw_data, path)

Saves the 3D voxel-wise data array to a JLD2 file.
"""
function save_voxel_wise_data(vw_data, path)
    jldsave(path; vw_data = vw_data)
end

"""
    read_volume(r, dir)

Reads a single NIfTI volume from the BIDS directory using the metadata in a measurements table row.
"""
function read_volume(r, dir)
    modality = hasproperty(r, :modality) ? r.modality : "anat"
    return niread(joinpath(dir, r.subject, r.session, modality, r.file)).raw
end

#= if contains_missings
    vw_data = Dict{Tuple{Int64, Int64, Int64}, Matrix{Union{Missing, number_type}}}()
else
    vw_data = Dict{Tuple{Int64, Int64, Int64}, Matrix{number_type}}()
end =#

#= function empty_data(contains_missings, number_type, n_row, n_col)
    if contains_missings
        data = missings(number_type, n_row, n_col)
    else
        data = fill(NaN, n_row, n_col)
    end
end =#




