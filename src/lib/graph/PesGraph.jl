module PesGraph

export
    create_pes_graph,
    eachnode, eacharc

using ..Utils
using ..Instance
using Memoization

struct Node
    t :: Int
    gamma :: Int
    received :: Int
end

function show_received(rcv::Int)
    if rcv == 0 return "∅" end
    
    return "{" * join(1:rcv, ",") * "}"
end

show_node(v::Node) = v.t < 0 ? "Term" :
    "Node($(v.t), $(v.gamma), $(show_received(v.received)))"
Base.show(io::IO, v::Node) = print(io, show_node(v))
Base.show(io::IO, vs::AbstractArray{Node}) =
    print(io, Utils.print_enum_array(vs))

struct Arc
    src :: Int # index of source node
    dst:: Int # index of destination node
end

show_arc(a::Arc) = "Arc($(a.src)->$(a.dst))"
Base.show(io::IO, a::Arc) = print(io, show_arc(a))
Base.show(io::IO, as::AbstractArray{Arc}) =
    print(io, Utils.print_array(as))

struct Graph
    nodes :: Vector{Node}
    arcs :: Vector{Arc}

    nodemap :: Dict{Node,Int}
    terminating :: Vector{Bool}
    iTerm :: Int
end

Base.show(io::IO, g::Graph) =
    print(io, "Nodes: ", g.nodes, "\nArcs: ", g.arcs)

@memoize function eachnode(g::Graph)
    return eachindex(g.nodes)
end

@memoize function eacharc(g::Graph)
    return eachindex(g.arcs)
end

# Gamma1 => number of deviating lead times
function compute_gamma1_usage(t, common_lead_time, deliveries)
    return sum((t != common_lead_time + delivery) ? 1 : 0 
        for delivery in deliveries; init=0)
end

# Gamma2 => sum of deviations
function compute_gamma2_usage(t, common_lead_time, deliveries)
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
    elseif type == Uncertainty.Gamma1
        return compute_gamma1_usage(t, inst.common_lead_time, deliveries)
    elseif type == Uncertainty.Gamma2
        return compute_gamma2_usage(t, inst.common_lead_time, deliveries)
    end

    throw("compute_budget_usage: Unhandled uncertainty type: $(type)")
end

@memoize function terminating_nodes(nodes, arcs, iTerm)
    terminating = [false for node in nodes]
    
    terminating[iTerm] = true
    queue = [iTerm]
    next = 1
    while next <= length(queue)
        inode = queue[next]

        for arc in arcs
            if arc.dst != inode continue end

            if !terminating[arc.src]
                terminating[arc.src] = true
                push!(queue, arc.src)
            end
        end

        next += 1
    end

    # for (node,term) in zip(nodes, terminating)
    #     if !term println("unterm:", node) end
    # end

    return terminating
end

function create_pes_graph(inst::LotSizingInstance):: Graph
    nodes = []
    arcs = []
    nodemap = Dict{Node,Int}()

    # Retrieve node index if present, else insert
    pushnode!(node) = get(nodemap, node) do
        push!(nodes, node)
        node_id = length(nodes)
        nodemap[node] = node_id

        return node_id
    end

    Term = Node(-1,0,0)
    iTerm = pushnode!(Term)

    Src = Node(inst.T, 0, 0)
    iSrc = pushnode!(Src)

    may_arrive = may_arrive_by(inst)
    must_arrive = must_arrive_by(inst)

    idx = 1
    t1 = inst.T # serves as info on max reached layer
    while idx+1 <= length(nodes)
        idx += 1
        node = nodes[idx]
        t1 = node.t + 1
        
        if t1 > inst.Tplus
            push!(arcs, Arc(idx, iTerm))
            continue
        end

        possible_received = (max(node.received,must_arrive[t1]) : may_arrive[t1])
        for next_received in possible_received
            next_gamma = node.gamma
            if !isnothing(inst.uncertainty.gamma)
                budget_usage = compute_budget_usage(inst, t1, node.received, next_received)

                next_gamma += budget_usage

                if next_gamma > inst.uncertainty.gamma
                    continue
                end
            end

            next_node = Node(t1, next_gamma, next_received)
            iNext = pushnode!(next_node)

            push!(arcs, Arc(idx, iNext))
        end
    end

    if t1 <= inst.Tplus
        println(t1)
        throw("PesGraph: Terminal node unreachable")
    end

    terminating = terminating_nodes(nodes, arcs, iTerm)

    arcs1 = [arc for arc in arcs if terminating[arc.dst]]

    return Graph(nodes, arcs1, nodemap, terminating, iTerm)
end

end # PesGraph
