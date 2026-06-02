__precompile__(true)
module PseudoPoly

import ..Utils
import ..PseudoPolyGraph as PPG

using ..Instance

using JuMP
using OffsetArrays
using ProgressMeter: @showprogress


export create_pseudopoly_algorithm

"""
Check assumptions required to construct pseudopolynomial algorithm. Throws an exception if assumption is not satisfied.

Arguments:
* inst - LotSizingInstance
"""
function assert_assumptions(inst::LotSizingInstance)
    # Check if uncertainty type is supported
    supported_uncertainty = [
        Uncertainty.GammaC,
        Uncertainty.GammaD,
    ]
    if (!(inst.uncertainty.type in supported_uncertainty))
        throw("Unsupported uncertainty type $(inst.uncertainty.type)")
    end

    # Integer capacity
    for (t,c) in enumerate(inst.production_capacity)
        if !isinteger(c)
            throw("Production capacity in period $(t) is not integer: $(c)")
        end
    end

    # Integer demand
    for (t,d) in enumerate(inst.demand)
        if !isinteger(d)
            throw("Demand in period $(t+inst.T) is not integer: $(d)")
        end
    end
end #assert_assumptions

function compute_arc_costs(
    inst::LotSizingInstance,
    graph::PPG.Graph
    ;
    make_optimizer=Utils.default_optimizer,
)
    costs = zeros(length(graph.arcs))
    prod = [[] for _ in eachindex(graph.arcs)]

    cumdem = cumulative_demand(inst)

    @showprogress for (i,arc) in enumerate(graph.arcs)
        u = graph.nodes[arc.src]
        v = graph.nodes[arc.dst]
        model = Model(make_optimizer)

        if (arc.dst == graph.iTerm)
            costs[i] = 0
            continue
        end

        inventory = inst.inventory_cost[v.t-inst.T]*(v.production - cumdem[v.t-inst.T])
        backorder = inst.backordering_cost[v.t-inst.T]*(cumdem[v.t-inst.T] - v.production)

        costs[i] = max(inventory, backorder)
        
        delivery = v.production - u.production

        if (delivery == 0)
            continue
        end

        ts = u.received+1:v.received
        @variable(model, y[t=ts], Bin)
        @variable(model, 0 <= x[t=ts] <= inst.production_capacity[t])
        
        @constraint(model, [t=ts],
            x[t] <= inst.production_capacity[t]*y[t])

        @constraint(model, sum(x[t] for t in ts) == delivery)

        @objective(model, Min,
            sum(inst.production_cost[t]*x[t] + inst.setup_cost[t]*y[t]
                for t in ts))

        optimize!(model)

        if !is_solved_and_feasible(model)
            throw("Cannot find costs for arc $(i): $(arc)")
        end

        costs[i] += objective_value(model)
        for t in ts
            push!(prod[i], (t,value(x[t])))
        end
    end

    return costs,prod
end

# GammaD => number of deviating lead times
function compute_gammaD_usage(t, common_lead_time, deliveries)
    return sum((t != common_lead_time + delivery) ? 1 : 0 
        for delivery in deliveries; init=0)
end

# GammaC => sum of deviations
function compute_gammaC_usage(t, common_lead_time, deliveries)
    return sum(abs(t - (common_lead_time + delivery))
        for delivery in deliveries; init=0)
end

function compute_budget_usage(
    inst::LotSizingInstance,
    t,
    curr_received, 
    next_received,
)
    deliveries = curr_received+1:next_received

    (;type) = inst.uncertainty
    if type == Uncertainty.Interval
        return 0
    elseif type == Uncertainty.GammaD
        return compute_gammaD_usage(t, inst.common_lead_time, deliveries)
    elseif type == Uncertainty.GammaC
        return compute_gammaC_usage(t, inst.common_lead_time, deliveries)
    end

    throw("compute_budget_usage: Unhandled uncertainty type: $(type)")
end

function compute_transition_times(inst::LotSizingInstance, graph::PPG.Graph)
    times = zeros(length(graph.arcs))

    for (i,arc) in enumerate(graph.arcs)
        u = graph.nodes[arc.src]
        v = graph.nodes[arc.dst]

        times[i] = compute_budget_usage(inst, v.t, u.received, v.received)
    end

    return times
end

"""
Solve Lot sizing problem using pseudopolynomial algorithm based on section (??) (TODO: fill the section number from final paper).

Arguments:
* inst - LotSizingInstance

Keyword arguments:
* make_optimizer - (default: default_optimizer) optimizer factory function
"""
function solve_pseudopoly(
    inst::LotSizingInstance,
    ;
    make_optimizer=Utils.default_optimizer,
)
    assert_assumptions(inst)

    graph = PPG.create_pseudopoly_graph(inst)

    arc_costs,prod = compute_arc_costs(inst, graph;
        make_optimizer=make_optimizer)

    times = compute_transition_times(inst, graph)

    N = length(graph.nodes)
    T = inst.uncertainty.gamma
    dp = OffsetArray(fill(Inf, N, T+1), 1:N, 0:T)
    pred = OffsetArray(fill((-1,0), N, T+1), 1:N, 0:T)

    node = graph.iSrc
    dp[node, 0] = 0
    queue = [(node,0)]
    i = 0
    while i < length(queue)
        i += 1
        (v, t) = queue[i]

        for iarc in graph.outstar[v]
            arc = graph.arcs[iarc]
            time = Int64(t + times[iarc])

            if time > T
                continue
            end

            cost = dp[v,t] + arc_costs[iarc]

            if cost < dp[arc.dst, time]
                dp[arc.dst, time] = cost

                push!(queue, (arc.dst, time))
                pred[arc.dst, time] = (iarc, t)
            end
        end
    end

    minprod = fill(Inf, 1:inst.T)
    mincost = Inf

    for t in 0:T
        cost = dp[graph.iTerm, t]
        if cost < mincost
            mincost = cost
            inode = graph.iTerm
            it = t
            while inode != graph.iSrc
                iarc,next_it = pred[inode,it]
                for (i,x) in prod[iarc]
                    minprod[i] = x
                end
                inode = graph.arcs[iarc].src
                it = next_it
            end
        end
    end

    if mincost == Inf
        throw("Problem is infeasible")
    end

    return mincost,minprod,[prod > 0 ? 1 : 0 for prod in minprod]
end

end #module PseudoPoly
