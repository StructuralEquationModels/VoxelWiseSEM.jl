function generate_coordinates(; mask)
    
    mask = isone.(niread(mask))

    x = Int[]
    y = Int[]
    z = Int[]
    voxel = Int[]

    for (i, ind) in enumerate(CartesianIndices(mask))
        if mask[ind]
            push!(x, ind[1])
            push!(y, ind[2])
            push!(z, ind[3])
            push!(voxel, i)
        end
    end

    coordinates = DataFrame(:voxel => voxel, :x => x, :y => y, :z => z)

    return coordinates
end