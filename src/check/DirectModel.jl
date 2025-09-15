import LotsizingModel as LM

import .LM.Instance
using .LM.Utils

using JuMP, Gurobi

function create_gurobi()
    env = Gurobi.Env(Dict{String,Any}(
        "OutputFlag" => 0
    ))

    return Gurobi.Optimizer(env)
end

function opt_model(inst::Instance.LotSizingInstance)
    model = JuMP.Model(create_gurobi)

    (;T,Tplus) = inst

    tprod = 1:T
    tfut = 1:Tplus-T

    M = sum(inst.production_capacity)

    @variable(model, 0 <= x[t=tprod] <= inst.production_capacity[t])
    @variable(model, y[tprod], Bin)

    @constraint(model, [t=tprod], x[t] <= M*y[t])

    @variable(model, B[tfut] >= 0)
    @variable(model, I[tfut] >= 0)
    
    D = cumsum(inst.demand)

    cS = inst.setup_cost
    cP = inst.production_cost
    cI = inst.inventory_cost
    cB = inst.backordering_cost

    @objective(model, Min,
        sum(y[t]*cS[t] + x[t]*cP[t] for t in tprod) +
        sum(I[t]*cI[t] + B[t]*cB[t] for t in tfut))
    
    # lead time modelling
    @variable(model, rho[p=tprod,f=tfut], Bin)

    LEAD = inst.common_lead_time

    earliest_arr(p) = p + LEAD - inst.early_deviation[p]
    latest_arr(p) = p + LEAD + inst.late_deviation[p]

    ## production arrives once
    @constraint(model, [p=tprod], sum(rho[p, f] for f in tfut) == 1)

    @constraint(model, [p=tprod,t=tfut; T+t<earliest_arr(p)], rho[p,t] == 0)
    @constraint(model, [p=tprod,t=tfut; T+t>latest_arr(p)], rho[p,t] == 0)

    @constraint(model, [p=skip_last(tprod),j=tfut,k=tfut; k < j &&
        (earliest_arr(p) <= T+j <= latest_arr(p)) &&
        (earliest_arr(p+1) <= T+k <= latest_arr(p+1))],
            rho[p,j] + rho[p+1,k] <= 1
    )

    @variable(model, fl[p=tprod,t=tfut] >= 0)
    @constraint(model, [p=tprod,t=tfut], fl[p,t] <= M*rho[p,t])
    @constraint(model, [p=tprod], sum(fl[p,t] for t in tfut) == x[p])

    X = [sum(fl[p,t1] for p in tprod, t1 in 1:t) for t in tfut]

    for t in tfut
        @constraint(model, B[t] - I[t] == D[t] - X[t])
    end

    JuMP.optimize!(model)

    println("Opt=$(JuMP.objective_value(model))")

    for t in Instance.prod_periods(inst)
        println("x[$(t)] => $(JuMP.value(x[t]))")
    end

    println("Far future backordering: $(JuMP.value.(B[Tplus-T]))")
end #opt_model

function prod_lead_cost_var(inst, model, x, y, zi, zb, arrival)
    (;T,Tplus) = inst
    tprod = 1:T
    tfut = 1:Tplus-T

    production = sum(inst.production_cost[t] * x[t] for t in tprod)
    setup = sum(inst.setup_cost[t] * y[t] for t in tprod)

    X = [sum(x[p] for p in tprod if arrival[p] <= t+T; init=0) 
        for t in tfut]
    
    D = Instance.cumulative_demand(inst)
    @constraint(model, [t=tfut], 
        zi[t] >= inst.inventory_cost[t]*(X[t] - D[t]))
    @constraint(model, [t=tfut], 
        zb[t] >= inst.backordering_cost[t]*(D[t] - X[t]))

    inventory = sum(zi[t] for t in tfut)
    backordering = sum(zb[t] for t in tfut)

    return production + setup + inventory + backordering
end #prod_lead_cost_var

function pes_model(inst::Instance.LotSizingInstance)
    model = JuMP.Model(create_gurobi)

    (;T,Tplus) = inst

    tprod = 1:T
    tfut = 1:Tplus-T
    
    @variable(model, 0 <= x[t=tprod] <= inst.production_capacity[t])
    @variable(model, y[tprod], Bin)
    
    M = sum(inst.production_capacity)
    @constraint(model, [t=tprod], x[t] <= M*y[t])

    scenarios = Instance.list_scenarios(inst)
    NS = length(scenarios)

    @variable(model, zmax >= 0)
    @variable(model, z[1:NS] >= 0)
    @variable(model, zi[1:NS,tfut] >= 0)
    @variable(model, zb[1:NS,tfut] >= 0)
    for (i,scen) in enumerate(scenarios)
        @constraint(model,
            prod_lead_cost_var(inst, model,
                x, y, zi[i,:], zb[i,:], scen) <= z[i])
        @constraint(model, z[i] <= zmax)
    end

    @objective(model, Min, zmax)

    print(model)

    JuMP.optimize!(model)

    println(JuMP.solution_summary(model))

    println("Pes=$(JuMP.objective_value(model))")

    for t in Instance.prod_periods(inst)
        println("x[$(t)] => $(JuMP.value(x[t]))")
    end
end

function run_direct_models(path)
    instance = open(path) do file
        return LM.Parse.parse_instance(file)
    end

    opt_model(instance)
end #run_direct_models
