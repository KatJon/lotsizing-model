__precompile__(true)
module Deterministic

import ..Utils
using ..Instance

using JuMP

function create_deterministic(
    inst::LotSizingInstance,
    ;
    make_optimizer = Utils.create_gurobi,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_attribute(model, "TimeLimit", time_limit)

    (;T,Tplus) = inst
    tprod = 1:T
    tfut = 1:Tplus-T

    @variable(model, y[t=tprod], Bin)
    @variable(model, 0 <= x[t=tprod] <= inst.production_capacity[t])

    @variable(model, B[t=tfut] >= 0)
    @variable(model, I[t=tfut] >= 0)

    M = sum(inst.production_capacity[t] for t in tprod)
    @constraint(model, [t=tprod], x[t] <= M*y[t])

    arrival = [t + inst.common_lead_time for t in tprod]

    X = [sum(x[p] for p in tprod if arrival[p] <= t+T; init=0) 
        for t in tfut]
    
    D = Instance.cumulative_demand(inst)

    @constraint(model, [t=tfut],
        B[t] - I[t] == D[t] - X[t])

    production = sum(inst.production_cost[t] * x[t] for t in tprod)
    setup = sum(inst.setup_cost[t] * y[t] for t in tprod)

    inventory = sum(inst.inventory_cost[t]*I[t] for t in tfut)
    backordering = sum(inst.backordering_cost[t]*B[t] for t in tfut)

    @objective(model, Min,
        production + setup + inventory + backordering)

    return model,x,y
end #create_deterministic

end #Deterministic
