__precompile__(true)
module MinMaxModel

using ..Instance
using ..Utils

import ..MinMaxGraph as PesG

using ..CommonModel

using JuMP

export create_minmax_model
export minmax_variables
export minmax_objective
export minmax_constraints

"""
Create minmax variables.

Arguments:
* inst - LotSizingInstance,
* model - JuMP model,
* graph - MinMaxGraph.

Returns:
    vector of variables z
"""
function minmax_variables(
    inst::LotSizingInstance,
    model, graph,
)
    # z_{v}
    @variable(model, z[v=PesG.eachnode(graph)] >= 0)

    return z
end #minmax_variables

"""
Create expression for objective in Min-Max model.

Arguments:
* inst - LotSizingInstance,
* graph - MinMaxGraph,
* z - vector of variables z,
* x - vector of production quantity variables,
* y - vector of setup variables.

Returns:
    Min-Max objective expression
"""
function minmax_objective(inst, graph, z, x, y)
    return (
        z[graph.iTerm] + 
        sum(
            x[t]*inst.production_cost[t] +
            y[t]*inst.setup_cost[t]
                for t in prod_periods(inst)
        )
    )
end #minmax_objective

"""
Add Min-Max constraints to `model`.

Arguments:
* inst - LotSizingInstance,
* model - JuMP model,
* graph - MinMaxGraph,
* z - vector of z variables,
* x - vector of production quantity variables.
"""
function minmax_constraints(inst, model, graph, z, x)
    # (35)
    @constraint(model, [v=PesG.eachnode(graph); graph.terminating[v] && v!=graph.iTerm],
        z[graph.iTerm] >= z[v])

    # (36) and (37)
    cum_demand = cumulative_demand(inst)
    for (;src,dst) in graph.arcs
        # exclude dummy arcs to terminal
        if dst == graph.iTerm continue end

        nDst = graph.nodes[dst]
        nSrc = graph.nodes[src]

        dst_received = sum(x[i] for i in 1:nDst.received; init=0)
        dst_demand = cum_demand[nDst.t - inst.T]

        @constraint(model,
            z[dst] - z[src] >= 
            inst.inventory_cost[nDst.t - inst.T] * 
            (dst_received - dst_demand)
        )

        @constraint(model,
            z[dst] - z[src] >= 
            inst.backordering_cost[nDst.t - inst.T] * 
            (dst_demand - dst_received)
        )
    end
end #minmax_constraints

"""
Create Min-Max model based on equations (?)-(?) TODO: UPDATE

Arguments:
* inst - LotSizingInstance.

Optional arguments:
* make_optimizer - (default: create_highs) optimizer factory function,
* time_limit - (default: 300) time limit in seconds.

Returns: tuple of
* model - JuMP model,
* x - vector of production quantity variables,
* y - vector of setup variables,
* z - vector of z variables,
* graph - MinMaxGraph.
"""
function create_minmax_model(
    inst::LotSizingInstance;
    make_optimizer = Utils.create_highs,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)
    
    graph = PesG.create_minmax_graph(inst)

    ## Variables
    x,y = common_variables(inst, model)
    z = minmax_variables(inst, model, graph)

    ## Objective
    @objective(model, Min,
        minmax_objective(inst, graph, z, x, y)
    )

    ## Constraints
    # (35)-(37)
    minmax_constraints(inst, model, graph, z, x)

    # (38)
    @constraint(model, [t=prod_periods(inst)],
        x[t] <= inst.production_capacity[t] * y[t])

    return model,x,y,z,graph
end #create_minmax_model

function pes_scenario_cost(
    inst::LotSizingInstance,
    xs::Vector{Float64},
    ;
    make_optimizer = Utils.create_highs,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_time_limit_sec(model, time_limit)
    
    ys = [x > 0 ? 1.0 : 0.0 for x in xs]

    graph = PesG.create_minmax_graph(inst)

    ## Variables
    z = minmax_variables(inst, model, graph)

    ## Objective
    @objective(model, Min,
        minmax_objective(inst, graph, z, xs, ys)
    )

    ## Constraints
    minmax_constraints(inst, model, graph, z, xs)

    return model
end #pes_scenario_cost

end #MinMaxModel
