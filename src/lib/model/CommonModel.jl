__precompile__(true)
module CommonModel

using ..Instance

using JuMP

export common_variables

"""
Create common variables for Lot Sizing models.

Arguments:
* inst - LotSizingInstance,
* model - JuMP model in which variables will be created.

Returns: tuple of
* x - vector of production amount variables
* y - vector of setup variables
"""
function common_variables(inst::LotSizingInstance, model)
    @variable(model,
        y[t=prod_periods(inst)], Bin)

    @variable(model,
        0 <= x[t=prod_periods(inst)] <= inst.production_capacity[t])

    return x,y
end #common_variables

end #CommonModel
