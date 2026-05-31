__precompile__(true)
module MipModel

using ..Instance
using ..Utils

using ..CommonModel

using JuMP

function time_periods(inst::LotSizingInstance)
    tprod = 1:inst.T
    tfut = 1:inst.Tplus-inst.T

    return tprod,tfut
end

function mip_minmin_variables(
    inst::LotSizingInstance,
    model::Model,
)
    tprod,tfut = time_periods(inst)

    @variable(model, B[tfut] >= 0)
    @variable(model, I[tfut] >= 0)
    @variable(model, rho[p=tprod,f=tfut], Bin)
    @variable(model, fl[p=tprod,t=tfut] >= 0)

    return B,I,rho,fl
end

function mip_minmin_constraints(
    inst::LotSizingInstance,
    model::Model,
    x, y, B, I, rho, fl,
)
    tprod,tfut = time_periods(inst)

    # lead time modelling
    LEAD = inst.common_lead_time
    earliest_arr(p) = p + LEAD - inst.early_deviation[p]
    latest_arr(p) = p + LEAD + inst.late_deviation[p]

    ## production arrives fully
    @constraint(model, [p=tprod], sum(rho[p, f] for f in tfut) == 1)

    (;T) = inst
    ## production cannot arrive before the earliest possible time
    @constraint(model, [p=tprod,t=tfut; T+t<earliest_arr(p)], rho[p,t] == 0)
    ## production cannot arrive after the latest possible time
    @constraint(model, [p=tprod,t=tfut; T+t>latest_arr(p)], rho[p,t] == 0)

    ## the arrivals may not cross-over (order of arrival is the same as order of production)
    (;T) = inst
    @constraint(model, [tp=skip_last(tprod),j=tfut,k=tfut; k < j &&
        (earliest_arr(tp) <= T+j <= latest_arr(tp)) &&
        (earliest_arr(tp+1) <= T+k <= latest_arr(tp+1))],
            rho[tp,j] + rho[tp+1,k] <= 1
    )

    # Uncertainty budget constraints
    ## GammaD
    if inst.uncertainty.type == Uncertainty.GammaD
        @constraint(model, 
            inst.uncertainty.gamma >= sum(rho[tp,tf-T] for
                tp in tprod,
                tf in earliest_arr(tp):latest_arr(tp)
                if tf != tp + inst.common_lead_time
            )
        )
    end

    ## GammaC
    if inst.uncertainty.type == Uncertainty.GammaC
        @constraint(model,
            inst.uncertainty.gamma >= sum(
                abs(tf - tp - inst.common_lead_time) * rho[tp,tf] for
                    tp in tprod,
                    tf in earliest_arr(tp):latest_arr(tp)
            )
        )
    end

    # production flows only in periods where it is arriving
    @constraint(model, [tp=tprod,tf=tfut], fl[tp,tf] <= inst.production_capacity[tp]*rho[tp,tf])

    # production flow is equal to production
    @constraint(model, [tp=tprod], sum(fl[tp,tf] for tf in tfut) == x[tp])

    X = [sum(fl[tp,t1] for tp in tprod, t1 in 1:tf) for tf in tfut]
    D = cumulative_demand(inst)
    for t in tfut
        # inventory balance constraint
        @constraint(model, B[t] - I[t] == D[t] - X[t])
    end
end

function mip_minmin_objective(
    inst::LotSizingInstance,
    x, y, B, I,
)
    tprod,tfut = time_periods(inst)

    cS = inst.setup_cost
    cP = inst.production_cost
    cI = inst.inventory_cost
    cB = inst.backordering_cost

    return (
        sum(y[t]*cS[t] + x[t]*cP[t] for t in tprod) +
        sum(I[t]*cI[t] + B[t]*cB[t] for t in tfut)
    )
end

"""
Create direct MIP model for the min-min problem.

Arguments:
* inst - LotSizingInstance
"""
function mip_minmin_model(inst::LotSizingInstance)
    model = JuMP.Model(create_highs)

    tprod,tfut = time_periods(inst)

    x,y = common_variables(inst, model)

    B,I,rho,fl = mip_minmin_variables(inst, model)

    # production can happen only after setup
    @constraint(model, [t=tprod], x[t] <= inst.production_capacity[t]*y[t])

    mip_minmin_constraints(inst, model, x, y, B, I, rho, fl)

    @objective(model, Min, mip_minmin_objective(inst, x, y, B, I))

    return model,x,y
end #mip_minmin_model

function opt_scenario_cost(
    inst::LotSizingInstance,
    xs::AbstractVector{Float64},
    ;
    make_optimizer = Utils.create_highs,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)

    ys = [(x > 0 ? 1.0 : 0.0) for x in xs]

    B,I,rho,fl = mip_minmin_variables(inst, model)

    mip_minmin_constraints(inst, model, xs, ys, B, I, rho, fl)

    @objective(model, Min, mip_minmin_objective(inst, xs, ys, B, I))

    return model
end

function prod_lead_cost_var(inst, model, x, y, zi, zb, arrival)
    (;T) = inst
    tprod,tfut = time_periods(inst)

    production = sum(inst.production_cost[t] * x[t] for t in tprod)
    setup = sum(inst.setup_cost[t] * y[t] for t in tprod)

    X = [sum(x[p] for p in tprod if arrival[p] <= t+T; init=0) 
        for t in tfut]
    
    D = cumulative_demand(inst)
    @constraint(model, [t=tfut], 
        zi[t] >= inst.inventory_cost[t]*(X[t] - D[t]))
    @constraint(model, [t=tfut], 
        zb[t] >= inst.backordering_cost[t]*(D[t] - X[t]))

    inventory = sum(zi[t] for t in tfut)
    backordering = sum(zb[t] for t in tfut)

    return production + setup + inventory + backordering
end #prod_lead_cost_var

function mip_minmax_variables(
    inst::LotSizingInstance,
    model::Model,
    NS::Int,
)
    _,tfut = time_periods(inst)

    @variable(model, zmax >= 0)
    @variable(model, z[1:NS] >= 0)
    @variable(model, zi[1:NS,tfut] >= 0)
    @variable(model, zb[1:NS,tfut] >= 0)

    return zmax,z,zi,zb
end #mip_minmax_variables

function mip_minmax_model(inst::LotSizingInstance)
    model = JuMP.Model(create_highs)

    tprod,tfut = time_periods(inst)
    
    x,y = common_variables(inst, model)
    
    @constraint(model, [t=tprod], x[t] <= inst.production_capacity[t]*y[t])

    scenarios = Instance.list_scenarios(inst)
    NS = length(scenarios)

    zmax,z,zi,zb = mip_minmax_variables(inst, model, NS)

    for (i,scen) in enumerate(scenarios)
        @constraint(model,
            prod_lead_cost_var(inst, model,
                x, y, zi[i,:], zb[i,:], scen) <= z[i])
        @constraint(model, z[i] <= zmax)
    end

    @objective(model, Min, zmax)

    return model,x,y
end

function pes_scenario_cost(
    inst::LotSizingInstance,
    xs::AbstractVector{Float64},
    ;
    make_optimizer = Utils.create_highs,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)

    ys = [(x > 0 ? 1.0 : 0.0) for x in xs]

    tprod,tfut = time_periods(inst)

    scenarios = Instance.list_scenarios(inst)
    NS = length(scenarios)

    zmax,z,zi,zb = mip_minmax_variables(inst, model, NS)

    for (i,scen) in enumerate(scenarios)
        @constraint(model,
            prod_lead_cost_var(inst, model,
                xs, ys, zi[i,:], zb[i,:], scen) <= z[i])
        @constraint(model, z[i] <= zmax)
    end

    @objective(model, Min, zmax)

    return model
end

function mip_R_model(
    inst::LotSizingInstance,
    ;
    make_optimizer = Utils.create_highs,
    time_limit = 5*60,
)
    if isnothing(inst.rbound)
        throw("Instance doesn't provide R* bound")
    end
    
    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)

    tprod,tfut = time_periods(inst)

    x,y = common_variables(inst, model)

    B,I,rho,fl = mip_minmin_variables(inst, model)

    # production can happen only after setup
    @constraint(model, [t=tprod], x[t] <= inst.production_capacity[t]*y[t])

    mip_minmin_constraints(inst, model, x, y, B, I, rho, fl)

    scenarios = Instance.list_scenarios(inst)
    NS = length(scenarios)

    zmax,z,zi,zb = mip_minmax_variables(inst, model, NS)

    for (i,scen) in enumerate(scenarios)
        @constraint(model,
            prod_lead_cost_var(inst, model,
                x, y, zi[i,:], zb[i,:], scen) <= z[i])
        @constraint(model, z[i] <= zmax)
    end

    @constraint(model, zmax <= inst.rbound)

    @objective(model, Min, mip_minmin_objective(inst, x, y, B, I))

    return model,x,y
end

function run_direct_models(path)
    instance = open(path) do file
        return LM.Parse.parse_instance(file)
    end

    opt_model(instance)
end #run_direct_models

end #MipModel
