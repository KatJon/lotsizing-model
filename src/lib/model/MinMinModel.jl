__precompile__(true)
module MinMinModel

using ..Instance
using ..Utils

import ..MinMinGraph as OptG

using ..CommonModel

using JuMP

export create_minmin_model
export minmin_variables
export minmin_constraints
export minmin_objective

"""
Create variables for Min-Min model

Arguments:
* inst - LotSizingInstance,
* model - model in which the variables are created,
* graph - MinMinGraph.

Returns: tuple of
* rho - array of binary variables denoting realization of the lead time,
* flow - array of continuous variables denoting flow between graph nodes.
"""
function minmin_variables(inst::LotSizingInstance, model, graph)
    @variable(model, 
        rho[t=prod_periods(inst), j=feasible_arrival_times(inst, t)], Bin)
    @variable(model,
        flow[u=OptG.eachnode(graph), v=OptG.eachnode(graph); OptG.is_arc(graph, u, v)] >= 0)

    return rho,flow
end #minmin_variables

"""
Create expression for objective in Min-Min model.

Arguments:
* inst - LotSizingInstance,
* graph - MinMinGraph,
* flow - array of flow variables,
* y - vector of y variables.

Returns:
Expression for objective in Min-Min model.
"""
function minmin_objective(
    inst::LotSizingInstance,
    graph, flow, y,
)
    arc_flow_cost(arc::OptG.Arc) = arc.cost * flow[arc.src,arc.dst]

    return (
        sum(arc_flow_cost(graph.arcs[iarc]) for iarc in OptG.eacharc(graph)) +
        sum(inst.setup_cost[t]*y[t] for t in prod_periods(inst))
    )
end #minmin_objective

"""
Add Min-Min constraints to model.

Arguments:
* inst - LotSizingInstance
* model - JuMP model
* graph - MinMinGraph
* rho - array of lead time realization variables
* flow - array of flow amount variables
* x - vector of production amount variables
"""
function minmin_constraints(
    inst::LotSizingInstance,
    model,
    graph,
    rho, flow, x,
)
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

    # (45) Gamma^d
    if inst.uncertainty.type == Uncertainty.GammaD
        @constraint(model,
            inst.uncertainty.gamma >= sum(rho[t,j] for
                t in prod_periods(inst),
                j in feasible_arrival_times(inst, t)
                if j != t + inst.common_lead_time
            ) 
        )
    # (45) Gamma^c
    elseif inst.uncertainty.type == Uncertainty.GammaC
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
end #minmin_constraints

"""
Create Min-Min model based on equations (?)-(?) TODO: update

Arguments:
* inst - LotSizingInstance.

Optional arguments:
* make_optimizer - (default: default_optimizer) optimizer factory function,
* time_limit - (default: 300) time limit in seconds.

Returns: tuple of
* model - JuMP model,
* x - vector of production quantity variables,
* y - vector of setup variables,
* rho - array of lead time realization variables,
* far_future_backordering - far future backordering quantity variable,
* flow - array of flow variables,
* graph - MinMinGraph.
"""
function create_minmin_model(
    inst::LotSizingInstance;
    make_optimizer = Utils.default_optimizer,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)

    graph = OptG.create_minmin_graph(inst)

    ## Variables
    x,y = common_variables(inst, model)
    rho,flow = minmin_variables(inst, model, graph)

    ## Objective
    @objective(model, Min,
        minmin_objective(inst, graph, flow, y))

    ## Constraints (43)-(48)
    minmin_constraints(inst, model, graph, rho, flow, x)

    # (49)
    @constraint(model, [t=prod_periods(inst)],
        x[t] <= inst.production_capacity[t] * y[t])
    
    far_future_backordering = flow[OptG.isnode(graph), OptG.ivnode(graph, inst.Tplus)]
    
    return model, x, y, rho, far_future_backordering, flow, graph
end #create_minmin_model

function opt_scenario_cost(
    inst::LotSizingInstance,
    xs::Vector{Float64},
    ;
    make_optimizer = Utils.default_optimizer,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)

    ys = [(x > 0 ? 1.0 : 0.0) for x in xs]

    graph = OptG.create_minmin_graph(inst)
    
    ## Variables
    rho,flow = minmin_variables(inst, model, graph)

    ## Objective
    @objective(model, Min,
        minmin_objective(inst, graph, flow, ys))

    minmin_constraints(inst, model, graph, rho, flow, xs)

    return model
end #opt_scenario_cost

get_opt_arrivals(inst::LotSizingInstance, rho) = [
    findfirstvalue(j -> value(rho[t,j]) == 1, feasible_arrival_times(inst, t)) 
        for t in prod_periods(inst)]

end #MinMinModel
