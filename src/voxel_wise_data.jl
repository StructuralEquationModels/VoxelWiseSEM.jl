"""
    voxel_wise_data(dir, measurements, coordinates) -> Array{Union{Missing, T}, 3}
 
 Reshapes BIDS brain volumes into a 3D array of voxel-wise data of size `(n_voxels, n_sessions, n_subjects)`
 
 Axis 1 (`n_voxels`) is indexed by the 1-based voxel index mapping from `coordinates.voxel_idx`
 Axis 2 (`n_sessions`) is indexed by `measurements.session_number`
 Axis 3 (`n_subjects`) is indexed by `measurements.subject_number`
 Missing subject/session combinations are filled with `missing`
 
 # Arguments
 - `dir`: path to the BIDS root directory
 - `measurements`: DataFrame returned by `generate_measurements`
 - `coordinates`: DataFrame returned by `generate_coordinates`
 
 # Example
 ```julia
 data = voxel_wise_data("/data/ds000224", measurements, coordinates)
 data[:, 1, 1]    # all voxels for session 1, subject 1
 data[1, :, :]    # all sessions and subjects for voxel index 1
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

    if !hasproperty(coordinates, :voxel_idx)
        coordinates.voxel_idx = 1:nrow(coordinates)
    end

    n_row = maximum(measurements.session_number)
    n_col = maximum(measurements.subject_number)
    n_voxels = nrow(coordinates)

    vw_data = missings(number_type, n_voxels, n_row, n_col)

    println("step 2/2:")
    @showprogress for (row_idx, c) in enumerate(eachrow(coordinates))
        x, y, z = c.x, c.y, c.z
        for (i, j) in keys(volumes)
            vw_data[row_idx, j, i] = volumes[(i, j)][x, y, z]
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

Reads a single NIfTI volume from the BIDS directory using the relative path stored in the measurements table row.
"""
function read_volume(r, dir)
    return niread(joinpath(dir, r.file)).raw
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




