module Hurwicz

using Memoization
using JuMP, Gurobi
using ..Utils
using ..Instance

using ..OptGraph
import ..OptGraph as OptG

using ..PesGraph
import ..PesGraph as PesG

import ..Model

export create_hurwicz_model

function create_hurwicz_model(
    inst::LotSizingInstance,
    alpha::Float64,
    ;
    make_optimizer = Utils.create_gurobi,
    time_limit = 5*60
)
    model = JuMP.Model(make_optimizer)
    set_attribute(model, "TimeLimit", time_limit)

    M = sum(inst.production_capacity[t] for t in prod_periods(inst))

    ograph = create_opt_graph(inst)
    pgraph = create_pes_graph(inst)

    ## Variables
    x,y = Model.create_common_variables(inst, model)
    rho,flow = Model.create_opt_variables(inst, model, ograph)
    z = Model.create_pes_variables(inst, model, pgraph)

    ## Objective
    @objective(model, Min,
       alpha*Model.pes_objective(inst, pgraph, z, x, y) +
       (1-alpha)*Model.opt_objective(inst, ograph, flow, y))

    ## Constraints

    Model.pes_constraints(inst, model, pgraph, z, x)

    Model.opt_constraints(inst, model, ograph, M, rho, flow, x)

    @constraint(model, [t=prod_periods(inst)],
        x[t] <= M * y[t])

    return model,x,y
end #create_hurwicz_model

end #Hurwicz
