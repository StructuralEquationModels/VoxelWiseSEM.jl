# get the task id when called from a slurm job array
slurm_array_id() = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

condition_filename(r) = join(reduce(vcat, [[var, string(r[var])] for var in names(r)]), "_")