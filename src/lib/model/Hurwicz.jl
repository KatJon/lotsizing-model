__precompile__(true)
module Hurwicz

using Memoization
using JuMP
using ..Utils
using ..Instance

import ..MinMinGraph as OptG
import ..MinMaxGraph as PesG

using ..CommonModel
import ..MinMinModel
import ..MinMaxModel
import ..Model

export create_hurwicz_model

"""
Create Hurwicz model for Lotsizing Problem.

Arguments:
* inst - LotSizingInstance,
* alpha - parameter for Hurwicz criterion.

Returns: tuple of
* model - JuMP model,
* x - vector of production quantity variables,
* y - vector of setup variables.
"""
function create_hurwicz_model(
    inst::LotSizingInstance,
    alpha::Float64,
    ;
    make_optimizer = Utils.default_optimizer,
    time_limit = 5*60
)
    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)

    M = sum(inst.production_capacity[t] for t in prod_periods(inst))

    ograph = OptG.create_minmin_graph(inst)
    pgraph = PesG.create_minmax_graph(inst)

    ## Variables
    x,y = common_variables(inst, model)
    rho,flow = MinMinModel.minmin_variables(inst, model, ograph)
    z = MinMaxModel.minmax_variables(inst, model, pgraph)

    ## Objective
    @objective(model, Min,
       alpha*MinMaxModel.minmax_objective(inst, pgraph, z, x, y) +
       (1-alpha)*MinMinModel.minmin_objective(inst, ograph, flow, y))

    ## Constraints

    MinMaxModel.minmax_constraints(inst, model, pgraph, z, x)

    MinMinModel.minmin_constraints(inst, model, ograph, rho, flow, x)

    @constraint(model, [t=prod_periods(inst)],
        x[t] <= M * y[t])

    return model,x,y
end #create_hurwicz_model

end #Hurwicz
