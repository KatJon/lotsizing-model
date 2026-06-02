__precompile__(true)
module SAA

import ..Utils
using ..Instance

import Random
using JuMP

export
    received_orders,
    create_SAA_model

"""
Calculate the vector of received quantities.

Arguments:
* inst - LotSizingInstance,
* arrivals - vector of arrival times for orders,
* x - vector of ordered quantities.

Returns: vector of received quantities.
"""
function received_orders(inst::LotSizingInstance, arrivals, x)
    (;T,Tplus) = inst
    tprod = 1:T
    tfut = 1:Tplus-T

    return [
        sum(x[p] for p in tprod if arrivals[p] <= t+T; init=0) 
        for t in tfut
    ]
end

"""
Build an SAA model for the instance based on the list of scenarios.

Arguments:
* inst - LotSizingInstance
* arrivals - vector of vectors of arrival times

Keyword arguments:
* make_optimizer - factory function for creating the JuMP optimizer 
    (default: default_optimizer)
* time_limit - time limit for solve in seconds (default: 5min (300s))

Returns: tuple of
* model - JuMP model
* x - vector of variables - production plan
* y - vector of variables - setup plan
"""
function create_SAA_model(
    inst::LotSizingInstance,
    arrivals::Vector{Vector{Int}},
    ;
    make_optimizer = Utils.default_optimizer,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)

    REPS = length(arrivals)
    scens = 1:REPS
    (;T,Tplus) = inst
    tprod = 1:T
    tfut = 1:Tplus-T

    @variable(model, y[t=tprod], Bin)

    @variable(model, 0 <= x[t=tprod] <= inst.production_capacity[t])

    @variable(model, zi[s=scens,t=tfut] >= 0)
    @variable(model, zb[s=scens,t=tfut] >= 0)
    
    M = sum(inst.production_capacity)
    @constraint(model, [t=tprod], x[t] <= M*y[t])

    D = Instance.cumulative_demand(inst)
    for s in scens
        Xs = received_orders(inst, arrivals[s], x)
        @constraint(model, [t=tfut], 
            zi[s,t] >= inst.inventory_cost[t]*(Xs[t] - D[t]))
        @constraint(model, [t=tfut], 
            zb[s,t] >= inst.backordering_cost[t]*(D[t] - Xs[t]))
    end #for

    inventory = sum(zi[s,t] for s in scens for t in tfut)
    backordering = sum(zb[s,t] for s in scens for t in tfut)

    production = sum(inst.production_cost[t] * x[t] for t in tprod)
    setup = sum(inst.setup_cost[t] * y[t] for t in tprod)

    @objective(model, Min,
        production + setup + (inventory + backordering)/REPS)
    
    return model, x, y
end

end #module SAA
