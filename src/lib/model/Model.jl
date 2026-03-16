__precompile__(true)
module Model

using Memoization
using JuMP, Gurobi
using ..Utils
using ..Instance

using ..OptGraph
import ..OptGraph as OptG

using ..PesGraph
import ..PesGraph as PesG

export create_opt_model, create_pes_model, create_R_model

function findfirstvalue(pred, itr)
    idx = findfirst(pred, itr)

    if isnothing(idx)
        return nothing
    end

    return itr[idx]
end #findfirstvalue

get_opt_arrivals(inst::LotSizingInstance, rho) = [
    findfirstvalue(j -> value(rho[t,j]) == 1, feasible_arrival_times(inst, t)) 
        for t in prod_periods(inst)]

function create_common_variables(inst, model)
    @variable(model,
        y[t=prod_periods(inst)], Bin)

    @variable(model,
        0 <= x[t=prod_periods(inst)] <= inst.production_capacity[t])

    return x,y
end #create_common_variables

function create_opt_variables(inst, model, graph)
    @variable(model, 
        rho[t=prod_periods(inst), j=feasible_arrival_times(inst, t)], Bin)
    @variable(model,
        flow[u=OptG.eachnode(graph), v=OptG.eachnode(graph); OptGraph.is_arc(graph, u, v)] >= 0)

    return rho,flow
end #create_opt_variables

function opt_objective(inst, graph, flow, y)
    arc_flow_cost(arc::OptGraph.Arc) = arc.cost * flow[arc.src,arc.dst]

    return (
        sum(arc_flow_cost(graph.arcs[iarc]) for iarc in OptG.eacharc(graph)) +
        sum(inst.setup_cost[t]*y[t] for t in prod_periods(inst))
    )
end #opt_objective

function opt_constraints(inst, model, graph, M, rho, flow, x)
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
        flow[isnode(graph), iunode(graph, t)] == x[t])

    # (47)
    for t in prod_periods(inst)
        for j in feasible_arrival_times(inst, t)
            @constraint(model,
                flow[iunode(graph, t), ivnode(graph, j)] <= rho[t,j] * M)
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
        instar = OptGraph.in_star(graph, i)
        outstar = OptGraph.out_star(graph, i)

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

    M = sum(inst.production_capacity[t] for t in prod_periods(inst))

    graph = create_opt_graph(inst)

    ## Variables
    x,y = create_common_variables(inst, model)
    rho,flow = create_opt_variables(inst, model, graph)

    ## Objective
    @objective(model, Min,
        opt_objective(inst, graph, flow, y))

    ## Constraints (43)-(48)
    opt_constraints(inst, model, graph, M, rho, flow, x)

    # (49)
    @constraint(model, [t=prod_periods(inst)],
        x[t] <= M * y[t])
    
    far_future_backordering = flow[isnode(graph), ivnode(graph, inst.Tplus)]
    
    return model, x, rho, far_future_backordering, flow, y, graph
end #create_opt_model

function create_pes_variables(inst, model, graph)
    # z_{v}
    @variable(model, z[v=PesG.eachnode(graph)] >= 0)

    return z
end #create_pes_variables

function pes_objective(inst, graph, z, x, y)
    return (
        z[graph.iTerm] + 
        sum(
            x[t]*inst.production_cost[t] +
            y[t]*inst.setup_cost[t]
            for t in prod_periods(inst)
        )
    )
end #pes_objective

function pes_constraints(inst, model, graph, z, x)
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
end #pes_constraints

function create_pes_model(
    inst::LotSizingInstance;
    make_optimizer = Utils.create_gurobi,
    time_limit = 5*60,
)
    model = JuMP.Model(make_optimizer)
    set_attribute(model, "TimeLimit", time_limit)

    M = sum(inst.production_capacity[t] for t in prod_periods(inst))
    
    graph = create_pes_graph(inst)

    ## Variables
    x,y = create_common_variables(inst, model)
    z = create_pes_variables(inst, model, graph)

    ## Objective
    @objective(model, Min,
        pes_objective(inst, graph, z, x, y)
    )

    ## Constraints
    # (35)-(37)
    pes_constraints(inst, model, graph, z, x)

    # (38)
    @constraint(model, [t=prod_periods(inst)],
        x[t] <= M * y[t])

    return model,x,y,z,graph
end #create_pes_model

function create_R_model(
    inst::LotSizingInstance;
    make_optimizer = Utils.create_gurobi,
    time_limit = 5*60
)
    if isnothing(inst.rbound)
        throw("Instance doesn't provide R* bound")
    end
    model = JuMP.Model(make_optimizer)
    set_attribute(model, "TimeLimit", time_limit)

    M = sum(inst.production_capacity[t] for t in prod_periods(inst))

    ograph = create_opt_graph(inst)
    pgraph = create_pes_graph(inst)

    ## Variables
    x,y = create_common_variables(inst, model)
    rho,flow = create_opt_variables(inst, model, ograph)
    z = create_pes_variables(inst, model, pgraph)

    ## Objective
    @objective(model, Min,
        opt_objective(inst, ograph, flow, y))

    ## Constraints
    # (54)
    B = inst.rbound
    @constraint(model,
        pes_objective(inst, pgraph, z, x, y) <= B)

    # (35)-(37)
    pes_constraints(inst, model, pgraph, z, x)

    # (43)-(48)
    opt_constraints(inst, model, ograph, M, rho, flow, x)

    # (38) & (49)
    @constraint(model, [t=prod_periods(inst)],
        x[t] <= M * y[t])

    far_future_backordering = flow[isnode(ograph), ivnode(ograph, inst.Tplus)]

    return model,x,rho,z,y,pgraph,far_future_backordering
end

end # module
