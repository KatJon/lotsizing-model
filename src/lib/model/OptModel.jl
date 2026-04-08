__precompile__(true)
module OptModel

using ..Instance
using ..Utils

import ..OptGraph as OptG

using ..CommonModel

using ..OptModel

using JuMP, Gurobi

export create_opt_model

get_opt_arrivals(inst::LotSizingInstance, rho) = [
    findfirstvalue(j -> value(rho[t,j]) == 1, feasible_arrival_times(inst, t)) 
        for t in prod_periods(inst)]

function create_opt_variables(inst, model, graph)
    @variable(model, 
        rho[t=prod_periods(inst), j=feasible_arrival_times(inst, t)], Bin)
    @variable(model,
        flow[u=OptG.eachnode(graph), v=OptG.eachnode(graph); OptG.is_arc(graph, u, v)] >= 0)

    return rho,flow
end #create_opt_variables

function opt_objective(inst, graph, flow, y)
    arc_flow_cost(arc::OptG.Arc) = arc.cost * flow[arc.src,arc.dst]

    return (
        sum(arc_flow_cost(graph.arcs[iarc]) for iarc in OptG.eacharc(graph)) +
        sum(inst.setup_cost[t]*y[t] for t in prod_periods(inst))
    )
end #opt_objective

function opt_constraints(inst, model, graph, rho, flow, x)
    # (43)
    @constraint(model, [t=prod_periods(inst)],
        sum(rho[t,j] for j in feasible_arrival_times(inst, t)) == 1)

    # (44)
    @constraint(model, [
            t=skip_last(prod_periods(inst)), 
            j=feasible_arrival_times(inst, t),
            k=feasible_arrival_times(inst, t+1);
            k < j
        ],
        rho[t,j] + rho[t+1, k] <= 1
    )

    # (45) Gamma 1
    if inst.uncertainty.type == Uncertainty.Gamma1
        @constraint(model,
            inst.uncertainty.gamma >= sum(rho[t,j] for
                t in prod_periods(inst),
                j in feasible_arrival_times(inst, t)
                if j != t + inst.common_lead_time
            ) 
        )
    # (45) Gamma 2
    elseif inst.uncertainty.type == Uncertainty.Gamma2
        @constraint(model,
            inst.uncertainty.gamma >= sum(
                abs(j - t - inst.common_lead_time) * rho[t,j] for
                    t in prod_periods(inst),
                    j in feasible_arrival_times(inst, t)
            )
        )
    end

    # (46)
    @constraint(model, [t=prod_periods(inst)],
        flow[OptG.isnode(graph), OptG.iunode(graph, t)] == x[t])

    # (47)
    for t in prod_periods(inst)
        for j in feasible_arrival_times(inst, t)
            u = OptG.iunode(graph, t)
            v = OptG.ivnode(graph, j)
            @constraint(model,
                flow[u,v] <= rho[t,j] * inst.production_capacity[t])
        end
    end

    # (48)
    ## capacity
    for i in OptG.eacharc(graph)
        arc = graph.arcs[i]
        @constraint(model,
           flow[arc.src, arc.dst] <= arc.capacity)
    end

    ## flow preservation
    for i in OptG.eachnode(graph)
        instar = OptG.in_star(graph, i)
        outstar = OptG.out_star(graph, i)

        @constraint(model,
            # incoming flow
            sum(flow[u, i] for u in instar) +
            # outgoing flow
            sum(-flow[i, v] for v in outstar) +
            # node supply
            graph.nodes[i].supply
            >= 0
        )
    end
end #opt_constraints

function create_opt_model(
    inst::LotSizingInstance;
    make_optimizer = Utils.create_gurobi,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_attribute(model, "TimeLimit", time_limit)

    graph = OptG.create_opt_graph(inst)

    ## Variables
    x,y = create_common_variables(inst, model)
    rho,flow = create_opt_variables(inst, model, graph)

    ## Objective
    @objective(model, Min,
        opt_objective(inst, graph, flow, y))

    ## Constraints (43)-(48)
    opt_constraints(inst, model, graph, rho, flow, x)

    # (49)
    @constraint(model, [t=prod_periods(inst)],
        x[t] <= inst.production_capacity[t] * y[t])
    
    far_future_backordering = flow[OptG.isnode(graph), OptG.ivnode(graph, inst.Tplus)]
    
    return model, x, y, rho, far_future_backordering, flow, graph
end #create_opt_model

function opt_scenario_cost(
    inst::LotSizingInstance,
    xs::Vector{Float64},
    ;
    make_optimizer = Utils.create_gurobi,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_attribute(model, "TimeLimit", time_limit)

    ys = [(x > 0 ? 1.0 : 0.0) for x in xs]

    graph = OptG.create_opt_graph(inst)
    
    ## Variables
    rho,flow = create_opt_variables(inst, model, graph)

    ## Objective
    @objective(model, Min,
        opt_objective(inst, graph, flow, ys))

    opt_constraints(inst, model, graph, rho, flow, xs)

    return model
end #opt_scenario_cost

end #OptModel
