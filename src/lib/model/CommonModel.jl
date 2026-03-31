__precompile__(true)
module CommonModel

using ..Instance

using JuMP

export create_common_variables

function create_common_variables(inst, model)
    @variable(model,
        y[t=prod_periods(inst)], Bin)

    @variable(model,
        0 <= x[t=prod_periods(inst)] <= inst.production_capacity[t])

    return x,y
end #create_common_variables

end #CommonModel
