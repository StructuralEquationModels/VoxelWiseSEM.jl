function generate_coordinates(; mask)
    
    mask = isone.(niread(mask))

    x = Int64[]
    y = Int64[]
    z = Int64[]

    for ind in CartesianIndices(mask)
        if mask[ind]
            push!(x, ind[1])
            push!(y, ind[2])
            push!(z, ind[3])
        end
    end

    coordinates = DataFrame(:x => x, :y => y, :z => z)

    return coordinates
end