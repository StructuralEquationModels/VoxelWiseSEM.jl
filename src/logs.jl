mutable struct PreProcLog
    steps::Vector
    logs::Vector
end

function PreProcLog()
    return PreProcLog([], [])
end

function Base.show(io::IO, log::PreProcLog)
    for i in 1:length(log.steps)
        print(io, "Step $i: \n")
        print(io, "\t")
        print(io, log.steps[i])
        print(io, "\n")
        print(io, "\t")
        print(io, log.logs[i])
        print(io, "\n")
    end
end

function Base.push!(log::PreProcLog, k, v)
    push!(log.steps, "record_"*k)
    push!(log.logs, Dict(k => v))
    return nothing
end

function Base.getindex(log::PreProcLog, s)
    return log.logs[s]
end

function save_log(log::PreProcLog, r)
    jldsave("logs/"*condition_filename(r)*".jld2", log = log)
end

function load_log(r)
    load("logs/"*condition_filename(r)*".jld2")["log"]
end