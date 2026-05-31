__precompile__(true)
module Hurwicz

using Memoization
using JuMP
using ..Utils
using ..Instance

import ..MinMinGraph as OptG
import ..MinMaxGraph as PesG

using ..CommonModel
import ..OptModel
import ..PesModel
import ..Model

export create_hurwicz_model

function create_hurwicz_model(
    inst::LotSizingInstance,
    alpha::Float64,
    ;
    make_optimizer = Utils.create_highs,
    time_limit = 5*60
)
    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)

    M = sum(inst.production_capacity[t] for t in prod_periods(inst))

    ograph = OptG.create_minmin_graph(inst)
    pgraph = PesG.create_minmax_graph(inst)

    ## Variables
    x,y = create_common_variables(inst, model)
    rho,flow = OptModel.create_opt_variables(inst, model, ograph)
    z = PesModel.create_pes_variables(inst, model, pgraph)

    ## Objective
    @objective(model, Min,
       alpha*PesModel.pes_objective(inst, pgraph, z, x, y) +
       (1-alpha)*OptModel.opt_objective(inst, ograph, flow, y))

    ## Constraints

    PesModel.pes_constraints(inst, model, pgraph, z, x)

    OptModel.opt_constraints(inst, model, ograph, rho, flow, x)

    @constraint(model, [t=prod_periods(inst)],
        x[t] <= M * y[t])

    return model,x,y
end #create_hurwicz_model

end #Hurwicz
