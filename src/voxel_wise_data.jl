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

    vw_data = Dict{String, Matrix{Union{Missing, number_type}}}()

    println("step 2/2:")
    @showprogress for c in eachrow(coordinates)
        x, y, z = c.x, c.y, c.z
        data = missings(number_type, n_row, n_col)
        for (i, j) in keys(volumes)
            data[i, j] = volumes[(i, j)][x, y, z]
        end
        push!(vw_data, string((x, y, z)) => data)
    end

    return vw_data
end

function save_voxel_wise_data(vw_data, path)
    f = jldopen(path, "w")
    @showprogress for v in keys(vw_data)
        write(f, string(v), vw_data[v])
    end
    close(f)
end

# helper
function read_volume(r, dir)
    return niread(joinpath(dir, r.subject, r.session, "anat", r.file)).raw
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




