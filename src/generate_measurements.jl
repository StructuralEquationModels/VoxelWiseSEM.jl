function generate_measurements(;dir)
    # find subjects
    subjects = readdir(dir)
    subjects = filter(x -> startswith(x, "sub"), subjects)
    # find sessions
    sessions = [find_sessions(dir, sub) for sub in subjects]
    # find files
    files = [[readdir(joinpath(dir, sub, ses, "anat")) for ses in sessions[i]] for (i, sub) in enumerate(subjects)]
    # put everything together in a DataFrame
    rows = []
    for (i, sub) in enumerate(subjects)
        for (j, ses) in enumerate(sessions[i])
            for (k, file) in enumerate(files[i][j])
                session_number = parse(Int, ses[5:end])
                push!(
                    rows, 
                    (
                        subject = sub, 
                        subject_number = i, 
                        session = ses, 
                        session_number = session_number,
                        file = file
                    )
                )
            end
        end
    end
    rows = DataFrame(rows)
    println("number of subjects:", maximum(rows.subject_number))
    println("number of sessions:", maximum(rows.session_number))
    return rows
end

# helper
function find_sessions(dir, sub)
    sessions = readdir(joinpath(dir, sub))
    sessions = filter(x -> startswith(x, "ses"), sessions)
    sessions = filter(x -> contains_anat(dir, sub, x), sessions)
    return sessions
end

function contains_anat(dir, sub, ses)
    datatypes = readdir(joinpath(dir, sub, ses))
    return any(datatypes .== "anat")
end