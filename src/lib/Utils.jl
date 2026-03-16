__precompile__(true)
module Utils

import Gurobi

export skip_first, skip_last

"Return view of subarray first element."
skip_first(itr::AbstractArray) = @view itr[begin+1:end]

"Return view of subarray skipping last element."
skip_last(itr::AbstractArray) = @view itr[begin:end-1]

"Pretty print an array without type information."
print_array(as::AbstractArray) = "[" * join(as, ",") * "]"

"Pretty print an array as a mapping."
print_enum_array(as::AbstractArray) = "[" * 
    join(["$(i)=>$(v)" for (i,v) in enumerate(as)], ",") * 
"]"

"Create a silent gurobi optimizer"
function create_gurobi()
    env = Gurobi.Env(Dict{String,Any}(
        "OutputFlag" => 0
    ))

    return Gurobi.Optimizer(env)
end #create_gurobi
end #Utils
