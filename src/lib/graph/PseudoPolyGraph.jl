__precompile__(true)
module PseudoPolyGraph

using ..Instance
using ..Utils

using Memoization

export Node, Arc, Graph
export create_pseudopoly_graph

struct Node
    t :: Int
    received :: Int
    production :: Int
end

function show_received(rcv::Int)
    if rcv == 0 return "∅" end
    
    return "{" * join(1:rcv, ",") * "}"
end

show_node(v::Node) = v.t < 0 ? "Term" :
    "Node($(v.t), $(show_received(v.received)), $(v.production))"
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
    # List of ids of arcs going out of the node."
    outstar :: Dict{Int,Vector{Int}} 
    
    iSrc :: Int
    iTerm :: Int
end

Base.show(io::IO, g::Graph) =
    print(io, "Nodes: ", g.nodes, "\nArcs: ", g.arcs)

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

    return terminating
end

function create_pseudopoly_graph(inst::LotSizingInstance)
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
            max_production = sum(inst.production_capacity[t] 
                for t in node.received+1:next_received; init=0)

            for prod in node.production:(node.production+max_production)
                next_node = Node(t1, next_received, prod)
                iNext = pushnode!(next_node)
                push!(arcs, Arc(idx, iNext))
            end
        end
    end

    if t1 <= inst.Tplus
        println(t1)
        throw("MinMaxGraph: Terminal node unreachable")
    end

    terminating = terminating_nodes(nodes, arcs, iTerm)

    arcs1 = [arc for arc in arcs if terminating[arc.dst]]

    outstar = Dict{Int, Vector{Int}}()

    for (i,arc) in enumerate(arcs)

        list = get!(outstar, arc.src, [])

        push!(list, i)
    end

    outstar[iTerm] = []

    return Graph(nodes, arcs1, nodemap, terminating, outstar, iSrc, iTerm)
end

end