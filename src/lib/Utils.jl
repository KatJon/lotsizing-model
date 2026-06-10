__precompile__(true)
module Utils

import HiGHS
import MathOptInterface as MOI

export skip_first, skip_last
export create_highs
export default_optimizer
export set_default_optimizer!
export findfirstvalue

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

"Create a silent highs optimizer"
function create_highs()
    optimizer = HiGHS.Optimizer()

    # Silence all possible HiGHS output
    MOI.set(optimizer, MOI.RawOptimizerAttribute("output_flag"), false)
    MOI.set(optimizer, MOI.RawOptimizerAttribute("log_to_console"), false)
    MOI.set(optimizer, MOI.RawOptimizerAttribute("log_dev_level"), 0)
    MOI.set(optimizer, MOI.RawOptimizerAttribute("mip_report_level"), 0)

    return optimizer
end #create_highs

"Default optimizer for models"
default_optimizer = create_highs

"Set default optimizer"
function set_default_optimizer!(optimizer)
    global default_optimizer = optimizer
end

"""
Find first value in `itr` satisfying predicate `pred`.
"""
function findfirstvalue(pred, itr)
    idx = findfirst(pred, itr)

    if isnothing(idx)
        return nothing
    end

    return itr[idx]
end #findfirstvalue

end #Utils
