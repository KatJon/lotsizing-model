module RModel
    
using ..Utils
using ..Instance

import ..MinMinGraph as OptG
import ..MinMaxGraph as PesG

using ..CommonModel
using ..OptModel
using ..PesModel

using Accessors: @set
using JuMP

"""
Create R* model for the instance.

Arguments:
* inst - LotSizingInstance

Optional arguments:
* make_optimizer - (default: create_highs) optimizer factory
* time_limit - (default: 5*60 s) time limit in seconds
* override_R_uncertainty - override uncertainty for the R* bound 
"""
function create_R_model(
    inst::LotSizingInstance;
    make_optimizer = Utils.create_highs,
    time_limit = 5*60,
    override_R_uncertainty = nothing
)
    if isnothing(inst.rbound)
        throw("Instance doesn't provide R* bound")
    end

    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)
    
    if isnothing(override_R_uncertainty)
        pinst = inst
    else
        pinst = @set inst.uncertainty=override_R_uncertainty
    end

    ograph = OptG.create_minmin_graph(inst)
    pgraph = PesG.create_minmax_graph(pinst)

    ## Variables
    x,y = create_common_variables(inst, model)
    rho,flow = OptModel.create_opt_variables(inst, model, ograph)
    z = PesModel.create_pes_variables(pinst, model, pgraph)

    ## Objective
    @objective(model, Min,
        OptModel.opt_objective(inst, ograph, flow, y))

    ## Constraints
    # (54)
    B = inst.rbound
    @constraint(model,
        PesModel.pes_objective(pinst, pgraph, z, x, y) <= B)

    # (35)-(37)
    PesModel.pes_constraints(pinst, model, pgraph, z, x)

    # (43)-(48)
    OptModel.opt_constraints(inst, model, ograph, rho, flow, x)

    # (38) & (49)
    @constraint(model, [t=prod_periods(inst)],
        x[t] <= inst.production_capacity[t] * y[t])

    far_future_backordering = flow[OptG.isnode(ograph), OptG.ivnode(ograph, inst.Tplus)]

    return model,x,y,rho,z,pgraph,far_future_backordering
end

"""
Optimize R model, and if not feasible solve the pessimistic model.

Arguments:
* inst - LotSizingInstance
* model - JuMP model
* x - vector of x variables
* y - vector of y variables
"""
function optimizeRmodel!(inst, model, x, y)
    optimize!(model)

    if is_solved_and_feasible(model)
        return model,x,y
    end

    pmodel,px,py = PesModel.create_pes_model(inst)

    optimize!(pmodel)

    if !is_solved_and_feasible(pmodel)
        throw("Pessimistic model for R* is infeasible")
    end

    return pmodel, px, py
end

end #RModel
