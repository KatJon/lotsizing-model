__precompile__(true)
module MinMinGraph

export
    create_minmin_graph,
    eachnode, eacharc,
    isnode, iunode, ivnode

using ..Utils
using ..Instance
using Memoization

struct Node
    name :: String
    supply :: Float64
end

show_node(v::Node) = "Node($(v.name), $(v.supply))"
Base.show(io::IO, v::Node) = print(io, show_node(v))
Base.show(io::IO, vs::AbstractArray{Node}) =
    print(io, Utils.print_array(vs))

struct Arc
    src :: Int # index of arc source
    dst :: Int # index of arc destination

    cost :: Float64
    capacity :: Float64
end

show_arc(a::Arc) = "Arc($(a.src)->$(a.dst), $(a.cost), $(a.capacity))"
Base.show(io::IO, a::Arc) = print(io, show_arc(a))
Base.show(io::IO, as::AbstractArray{Arc}) =
    print(io, Utils.print_array(as))

struct Graph
    nodes :: Vector{Node}
    arcs :: Vector{Arc}
    _unodes :: Int
    _vnodes :: Int
end

Base.show(io::IO, g::Graph) = print(io, "Nodes: ", g.nodes, "\nArcs: ", g.arcs)


@memoize function eachnode(g::Graph)
    return eachindex(g.nodes)
end

@memoize function eacharc(g::Graph)
    return eachindex(g.arcs)
end

"""
Get id of S node
"""
isnode(g::Graph) = 1

"""
Get id of node u_t, t in [T]
"""
iunode(g::Graph, t) = g._unodes + t

"""
Get id of node v_t, t in [T+]
"""
ivnode(g::Graph, t) = g._vnodes + t

@memoize function get_arc(g::Graph, isrc, idst)
    idx = findfirst(a -> a.src == isrc && a.dst == idst, g.arcs)

    if isnothing(idx)
        return nothing
    end

    return g.arcs[idx]
end

@memoize function is_arc(g::Graph, isrc, idst)
    return !isnothing(get_arc(g, isrc, idst))
end

@memoize function in_star(g::Graph, idst)
    return [g.arcs[i].src for i in eachindex(g.arcs)
        if g.arcs[i].dst == idst]
end

@memoize function out_star(g::Graph, isrc)
    return [g.arcs[i].dst for i in eachindex(g.arcs) 
        if g.arcs[i].src == isrc]
end

"""
Node order:
S,u_1,...,u_T,v_T+1,...,V_T+
"""
function create_minmin_graph(inst::LotSizingInstance)
    (; T, Tplus) = inst
    nodes = []
    arcs = []

    M = sum(inst.production_capacity[t] for t in prod_periods(inst))

    # Nodes
    ## S node
    source_supply = sum(inst.demand[t-T] for t in future_periods(inst); init=0)
    push!(nodes, Node("S", source_supply))

    S = length(nodes)

    ## u_t nodes
    unodes = length(nodes)

    for t in prod_periods(inst)
        push!(nodes, Node("u_$(t)", 0))
    end

    ## v_t nodes
    vnodes = length(nodes) - T
    
    for t in future_periods(inst)
        push!(nodes, Node("v_$(t)", -inst.demand[t-T]))
    end

    # Arcs
    ## Production arcs
    for t in prod_periods(inst)
        push!(arcs, 
            Arc(S, unodes + t,
                inst.production_cost[t], 
                inst.production_capacity[t]))
    end
    
    ## Lead-time arcs
    for t in prod_periods(inst)
        for l in feasible_arrival_times(inst, t)
            push!(arcs,
                Arc(unodes + t, vnodes + l, 0, M))
        end
    end

    ## Inventory and backordering arcs
    for t in skip_last(future_periods(inst))
        # Inventory
        push!(arcs,
            Arc(vnodes + t, vnodes + t + 1,
                inst.inventory_cost[t-T], M))
        
        # Backordering
        push!(arcs,
            Arc(vnodes + t + 1, vnodes + t,
                inst.backordering_cost[t-T], M))
    end

    ## Far future backordering arc
    push!(arcs,
        Arc(S, vnodes + Tplus,
            inst.backordering_cost[Tplus-T],
            sum(inst.demand)))

    return Graph(nodes, arcs, unodes, vnodes)
end #create_minmin_graph

end #MinMinGraph
