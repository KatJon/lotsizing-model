import LotsizingModel as LM

import .LM.Instance
import .LM.Instance.Uncertainty

using JuMP, Gurobi

using Printf
using OffsetArrays: Origin
using Random

"""
Generate an instance for partition problem.
Arguments:
* N - number of items
* k - maximal size of item
* rng - optional, random number generator instance

Returns a sorted vector of items
"""
function gen_partition(
    N::Int,
    k::Int,
    ;
    rng::AbstractRNG=Xoshiro()
)
    X = zeros(Int, N)
    while true
        rand!(rng, X, 1:k)

        if sum(X) % 2 == 0 break end
    end

    return sort(X)
end

# """
# Solve partition problem using pseudopolynomial algorithm. This version doesn't check for the balanced number of items in solution.

# Arguments:
# * A - vector of sorted items

# Preconditions:
# * A is nondecreasing

# Returns: binary vector of selected items
# """
# function solve_partition_basic(A::Vector{Int})
#     N = length(A)
#     if N % 2 == 1 throw("Number of items is not even") end

#     SUM = sum(A)
#     if SUM % 2 == 1 throw("Sum of data is not even") end
#     S = Int64(SUM/2)

#     store = Origin(0)(zeros(Bool, N+1, S+1))

#     for i in 0:N
#         for j in 0:S
#             if i == 0 store[i,j] = false end
#             if j == 0 store[i,j] = true end
#         end
#     end

#     for i in 1:N
#         for j in 1:S
#             if data[i] > j
#                 store[i,j] = store[i-1, j]
#             else
#                 inclusion = store[i-1,j-A[i]]
#                 exclusion = store[i-1,j]

#                 store[i,j] = inclusion || exclusion
#             end
#         end
#     end

#     return store[N,S]
# end

"""
Solve partition problem (with constraint on selecting half of the items) using pseudopolynomial algorithm.

Arguments:
* A - vector of sorted items

Preconditions:
* A is nondecreasing

Returns: tuple of
    * boolean flag denoting if the solution was found
    * vector of selected items or nothing if there is no solution
    * binary vector of selected items or nothing if there is no solution
"""
function solve_partition_balanced(A::Vector{Int})
    N = length(A)
    if N % 2 == 1 throw("Number of items is not even") end
    K = Int64(N / 2) # Size of partition

    SUM = sum(A)
    if SUM % 2 == 1 throw("Sum of data is not even") end
    S = Int64(SUM/2)

    # S[k][t] - can sum k numbers to t
    C = Origin(0)(zeros(Bool, K+1, S+1))
    C[0,0] = true

    parent = Origin(0)(fill((0,0), K+1, S+1))

    for a in A
        for k in reverse(1:K)
            for t in reverse(a:S)
                if C[k-1, t-a]
                    if !C[k,t] 
                        C[k,t] = true
                        parent[k,t] = (t-a,a)
                    end
                end
            end
        end
    end

    if !C[K,S] return false, nothing, nothing end

    SOL = zeros(Int, K)

    target = S
    for k in reverse(1:K)
        prev,x = parent[k,target]
        SOL[k] = x
        target = prev
    end


    idA = [( if !isnothing(findfirst(x->x==A[i], SOL)) 1 else 0 end) for (i,S) in zip(1:N,SOL)]

    return true, SOL, idA
end

"""
Create an R* LotSizing instance from the Partition instance.

Arguments:
* A - nondecreasing vector of items

Returns: R*LotSizing instance
"""
function build_instance_from_partition(A::Vector{Int})
    if !issorted(A) throw("Input array is assumed to be sorted") end

    # Instance built based on Theorem 1.
    S = Int(sum(A) / 2)
    N = length(A)
    # time periods
    T = N
    Tplus = 2*N + 1
    
    prods = 1:T
    futures = (T+1):Tplus
    zfutures = 1:(Tplus-T)

    # production params
    setup_cost = [A[t] for t in prods]
    production_cost = [0 for t in prods]
    production_capacity = [1 for t in prods]

    # lead time params
    common_lead_time = N
    late_deviation = [N+1 - t for t in prods]
    early_deviation = [0 for t in prods]

    # futures
    demand = zeros(Float64, Tplus-T)
    demand[Tplus-T] = Int64(N/2)

    inventory_cost = zeros(Float64, Tplus-T)
    for t in 1:N-1
        inventory_cost[t] = 2*A[t+1] - 2*A[t]
    end
    inventory_cost[N] = 2*S - 2*A[N]
    inventory_cost[N+1] = 0

    # "Infinite" backordering cost
    backordering_cost = [S*S for t in T:Tplus]

    # uncertainty modelling
    uncertainty = Uncertainty.Model(Uncertainty.Interval, nothing)

    # R* bound
    rbound = (N-1) * S

    return Instance.LotSizingInstance(
        T, Tplus,
        setup_cost,
        production_cost,
        production_capacity,
        common_lead_time,
        late_deviation,
        early_deviation,
        demand,
        inventory_cost,
        backordering_cost,
        uncertainty,
        rbound,
    )
end

function build_partition_model(env::Gurobi.Env, A::Vector{Int})
    mkGurobi = () -> Gurobi.Optimizer(env)

    instance = build_instance_from_partition(A)

    model,x,rho,_,_,_,far_future_backordering = LM.Model.create_R_model(instance,make_optimizer=mkGurobi)

    return instance,model,x,far_future_backordering,rho
end

function partition_from_rmodel(A, model)
    #TODO: fix objective by rounding (answers need to be integer)
    obj = JuMP.objective_value(model)

    half = Int64(sum(A) / 2)
    println("$(@sprintf("%.8f",obj-half))")
    return isapprox(obj, half)
end #partition_from_rmodel

function mkEnv()
    return Gurobi.Env(Dict{String,Any}(
        "OutputFlag" => 0
    ))
end

function test_single()
    env = mkEnv()

    N = 10
    A = gen_partition(N, 50)

    passing,ADP = solve_partition_balanced(A)


    println("Instance:")
    println(A)
    println("Answer (DP): $(passing)")

    inst,model,x,far_future_backordering,rho = build_partition_model(env, A)

    JuMP.optimize!(model)

    println("DP sol: $(ADP) (sum=$(sum(ADP)))")

    val = sum(JuMP.value.(x[t]) * A[t] for t in Instance.prod_periods(inst))

    for t in Instance.prod_periods(inst)
        println("$(t) $(JuMP.value(x[t])) $(A[t])")
    end

    println("S=$(sum(A)/2), solS=$(val)")

    partition_from_rmodel(A, model)
    println("Far future backordering: $(JuMP.value(far_future_backordering))")
end

"""
Conduct an experiment validating the correctness of the reduction from Partition problem.

Arguments:
* rep - number of repetitions (default 100)
* N - number of items in problem (defualt 10)
* size - maximal size of items (default N*N)

Returns: tuple of
* number of tests passing
* number of all tests
* number of yes-instances (based on DP solutions)
* number of no-instances
"""
function test_mult(;rep=100,N=10,size=N*N)
    env = mkEnv()
    yes = 0
    no = 0

    pass = 0

    for i in 1:rep
        A = gen_partition(N, size)
        okDP,ADP = solve_partition_balanced(A)

        _,model,x = build_partition_model(env, A)
        JuMP.optimize!(model)

        okR = partition_from_rmodel(A, model)

        if okDP yes += 1 else no += 1 end

        if okDP == okR
            pass += 1
        else
            println("*** ***")
            println(A)
            println([JuMP.value(x[i]) for i in 1:N])
            println("*** ***")
        end

        println("$(i) $(okDP ? 1 : 0) $(okR ? 1 : 0)")
    end

    println("Pass: $(pass) ($(round(100*pass/rep,digits=2))%)")

    println("Yes-instances: $(yes)")
    println("No-instances: $(no)")

    return pass, rep, yes, no
end
