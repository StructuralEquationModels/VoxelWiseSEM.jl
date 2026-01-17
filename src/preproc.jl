############################################################################################
# missings and zeros
"""
    step_missings!(coordinates::DataFrame, log::Dict, data::Union{Array, Dict})

Computes
- the number of non-missing data points per voxel as "n_nonmissing".
Removes
- voxels with only missings
Logs
- the number voxels with only missings as "all_missing".
"""
function step_missings!(coordinates::DataFrame, log::PreProcLog, data::Union{Array, Dict})
    push!(log.steps, "step_missings!")
    this_log = Dict()
    coordinates = apply_voxelwise(n_nonmissing, coordinates, data)
    this_log["all_missing"] = sum(iszero.(coordinates.n_nonmissing))
    filter!(r -> !iszero(r.n_nonmissing), coordinates)
    push!(log.logs, this_log)
    return coordinates
end

"""
    step_zeros!(coordinates::DataFrame, log::Dict, data::Union{Array, Dict})

Computes
- the proportion of zero values for each voxel as "p_zero".
Removes
- voxels with only zeros.
Logs
- the number of voxels containing only zeros as "all_zero".
- the number of voxels containing some zeros as "some_zero".
"""
function step_zeros!(coordinates::DataFrame, log::PreProcLog, data::Union{Array, Dict})
    push!(log.steps, "step_zeros!")
    this_log = Dict()
    coordinates = apply_voxelwise(p_zero, coordinates, data)
    all_zero = isone.(coordinates.p_zero)
    this_log["all_zero"] = sum(all_zero)
    this_log["some_zero"] = sum((.!all_zero) .& (coordinates.p_zero .> 0))
    filter!(r -> !isone(r.p_zero), coordinates)
    push!(log.logs, this_log)
    return coordinates
end

n_nonmissing(data) = (n_nonmissing = sum(.!ismissing.(data)),)

p_zero(data) = (p_zero = sum(iszero.(skipmissing(data))) / sum(.!ismissing.(data)),)

all_zero(data) = (all_zero = all(ismissing.(data) .| iszero.(data)),)

############################################################################################
# outliers
"""
    step_mad!(coordinates::DataFrame, log::Dict, data::Union{Array, Dict};
        mad_cutoff, kwargs...)

Computes 
- MAD (median absolute deviation) for each voxel as "mad".
    Per default, the MAD is scaled to the sd of a normal distribution -
    this can be turned off by passing the option to `kwargs...`.
Sets
- all values more than `mad_cutoff`*MAD away from the median to missing.
Removes 
- all voxels with a zero MAD.
Logs
- all voxels with a zero MAD as "mad_zero".
- the fraction of data removed in total as "removed_data_fraction".
- "mad_cutoff".

Input:
    "coordinates" needs to contain the column "n_nonmissing" containing the number of non-missing
    data points in each voxel before outlier removal (step_missings!).
"""
function step_mad!(coordinates::DataFrame, log::PreProcLog, data::Union{Array, Dict}; 
        mad_cutoff, kwargs...)
    push!(log.steps, "step_mad!")
    this_log = Dict()
    # compute mad and remove 
    this_log["mad_cutoff"] = mad_cutoff
    coordinates = apply_voxelwise(
        set_missing_mad!, coordinates, data; cutoff = mad_cutoff, kwargs...)
    this_log["mad_zero"] = sum(iszero.(coordinates.mad))
    this_log["removed_data_fraction"] = 
        sum(coordinates.n_outlier_removed)/sum(coordinates.n_nonmissing)
    filter!(r -> !iszero(r.mad), coordinates)
    push!(log.logs, this_log)
    return coordinates
end

"""
    step_rm_voxel!(coordinates::DataFrame, log::Dict, data::Union{Array, Dict}; voxel_cutoff)

Removes
- all voxels where a higher proportion than "voxel_cutoff" has been removed as outliers.
Logs
- the fraction of removed voxels als "removed_voxel_fraction".
- "voxel_cutoff".

Input:
    "coordinates" needs to contain columns "n_nonmissing" containing the number of non-missing
    data points in each voxel before outlier removal (step_missings!) and "n_outlier_removed" 
    containing the number of outliers removed in each voxel (step_mad!).
"""
function step_rm_voxel!(coordinates::DataFrame, log::PreProcLog, data::Union{Array, Dict}; 
        voxel_cutoff)
    push!(log.steps, "step_rm_voxel!")
    this_log = Dict()
    this_log["voxel_cutoff"] = voxel_cutoff
    this_log["removed_voxel_fraction"] = 
        sum(coordinates.n_outlier_removed .>= voxel_cutoff*coordinates.n_nonmissing)/
            size(coordinates, 1)
    filter!(r -> (r.n_outlier_removed < voxel_cutoff*r.n_nonmissing), coordinates)
    push!(log.logs, this_log)
    return coordinates
end

compute_mad(data; kwargs...) = (mad = mad(skipmissing(data); kwargs...),)

function set_missing_mad!(data; cutoff, kwargs...)
    m = mad(skipmissing(data), kwargs...)
    if !iszero(m)
        deviation = 
            abs.((data .- median(skipmissing(data)))./m)
        ind_mad = (deviation .> cutoff)
        n_removed = sum(skipmissing(ind_mad))
        data[ind_mad .| ismissing.(data)] .= missing
    else
        n_removed = sum(.!ismissing.(data))
        data .= missing
    end
    return (n_outlier_removed = n_removed, mad = m,)
end