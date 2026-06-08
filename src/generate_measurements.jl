"""
    generate_measurements(;dir, modality = "anat") -> DataFrame
 
Scans a BIDS dataset and returns a DataFrame with one row per NIfTI file found.
Only '.nii' and '.nii.gz' files are included.
 
# Arguments
- 'dir': path to the BIDS root directory.
- 'modality': the BIDS datatype subfolder to look in ('"anat"', '"func"', '"dwi"', …).
 
# Returns
A DataFrame with columns:
- 'subject': BIDS subject label, e.g. '"sub-MSC01"'
- 'subject_number': integer index
- 'session': BIDS session label, e.g. '"ses-struct01"'
- 'session_number': integer parsed from the trailing digits of the session label
- 'modality': the datatype folder name, e.g. '"anat"'
- 'file': filename of the NIfTI file
"""
function generate_measurements(;dir, modality = "anat")
    # find subjects
    subjects = readdir(dir)
    subjects = sort(filter(x -> startswith(x, "sub-"), subjects))  # the trailing dash avoids false matches
    # sort() is added so subject_number is always assigned in alphabetical order,
    # making the numbering deterministic regardless of filesystem order.

    # find sessions
    sessions = [find_sessions(dir, sub, modality) for sub in subjects]
    # modality is now forwarded to find_sessions so the function works
    # for any BIDS datatype folder ("anat", "func", "dwi", …), not just "anat"

    # find files
    files = [[filter(x -> endswith(x, ".nii") || endswith(x, ".nii.gz"), readdir(joinpath(dir, sub, ses, modality))) 
    for ses in sessions[i]] for (i, sub) in enumerate(subjects)]
    # put everything together in a DataFrame


    rows = []
    for (i, sub) in enumerate(subjects)
        for (j, ses) in enumerate(sessions[i])
            for (k, file) in enumerate(files[i][j])
                session_number = _parse_session_number(ses)
                # That hard-coded slice breaks for session labels like "ses-func01" or "ses-baseline"
                # _parse_session_number extracts the trailing digits robustly.
                push!(
                    rows, 
                    (
                        subject = sub, 
                        subject_number = i, 
                        session = ses, 
                        session_number = session_number,
                        modality= modality,
                        file = file
                    )
                )
            end
        end
    end
    rows = DataFrame(rows)
    println("number of subjects:", unique(rows.subject_number))
    println("number of sessions:", unique(rows.session_number)) 
    # maximum() would crash on an empty DataFrame; unique() is safe and
    # also more accurate (counts distinct session labels, not the highest index)
    return rows
end

"""
    save_measurements(measurements, path)
 
Writes the measurements DataFrame to a CSV file at `path`.
Load it back with `CSV.read(path, DataFrame)`.
 
# Example
```julia
save_measurements(measurements, "measurements.csv")
measurements = CSV.read("measurements.csv", DataFrame)
```
"""
function save_measurements(measurements::DataFrame, path::AbstractString)
    CSV.write(path, measurements)
    println("measurements saved to \"", path, "\"")
end

# helper
"""
    find_sessions(dir, sub, modality) -> Vector{String}
 
Returns a sorted list of session folder names for `sub` that contain the
given `modality` subfolder.
"""
function find_sessions(dir, sub, modality)
    sessions = readdir(joinpath(dir, sub))
    sessions = filter(x -> startswith(x, "ses-"), sessions)
    sessions = filter(x -> contains_modality(dir, sub, x, modality), sessions)
    return sort(sessions)
end

#helper for modality
"""
    contains_modality(dir, sub, ses, modality) -> Bool
 
Returns `true` if the session folder contains a subfolder named `modality`
"""
function contains_modality(dir, sub, ses, modality)
    datatypes = readdir(joinpath(dir, sub, ses))
    return any(datatypes .== modality)
end

"""
    _parse_session_number(ses) -> Int
 
Extracts the trailing integer from a BIDS session label.
Returns `0` if no digits are found.
 
# Examples
```julia
_parse_session_number("ses-01")       # → 1
_parse_session_number("ses-struct02") # → 2
```
"""
function _parse_session_number(ses::AbstractString)
    m = match(r"\d+$", ses)  
    if m === nothing
        return 0
    else
        return parse(Int, m.match)
    end
end