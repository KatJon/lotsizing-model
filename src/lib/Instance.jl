__precompile__(true)
module Instance

using ..Utils

using ProgressMeter
using Memoization

import Random
using Base.Threads

export LotSizingInstance
export Uncertainty
export
    cumulative_demand,
    feasible_arrival_times,
    feasible_lead_times,
    future_periods,
    lead_max,
    lead_min,
    max_gammaC_budget,
    may_arrive_by,
    must_arrive_by,
    prod_lead_cost,
    prod_periods,
    sample_arrivals
export instance_sampling_weights

"""
Provides type definitions for uncertainty modelling
"""
module Uncertainty
    """
    Enumeration type describing uncertainty set:
    - `None`: no uncertainty -- deterministic scenario
    - `Interval`: interval uncertainty -- all possible lead-time deviations are allowed.
    - `GammaD`: interval uncertainty with budget constraint on the number of deviations
    - `GammaC`: interval uncertainty with budget constraint on the sum of deviations
    """
    @enum Type None Interval GammaD GammaC

    """
    Structure holding description of the uncertainty set.
    - `type`: uncertainty type.
    - `gamma`: for budgeted uncertainty types (GammaD, GammaC) provides the budget ``\\Gamma``, for other types it should be `nothing`.
    """
    struct Model
        type :: Type
        gamma :: Union{Nothing, Int64}
    end

    function show(unc::Model)::String
        if unc.type == None
            return "NONE"
        elseif unc.type == Interval
            return "INTERVAL"
        elseif unc.type == GammaD
            return "GAMMAD $(unc.gamma)"
        elseif unc.type == GammaC
            return "GAMMAC $(unc.gamma)"
        else
            throw("Unhandled uncertainty type $(unc.type)")
        end
    end
end #Uncertainty

"""
Structure `LotSizingInstance` holds the input data describing the instance of
(R*) Lot-sizing Problem with Lead Time Uncertainty.

# Fields
## Time Periods
- `T` -- (``T``) number of production periods 1,...,T.
- `Tplus` -- (``T^+``) number of all periods, future periods are the periods T+1,...,Tplus.
## Production period parameters
- `setup_cost` -- (``c^S_t``) cost of starting production in production period `t`, t <- 1,...,T.
- `production_cost` -- (``c^P_t``) unit cost of production in production period `t`, t <- 1,...,T.
- `production_capacity` -- (``b_t``) maximal capacity for production in period `t`, t <- 1,...,T.
## Lead time characterization
- `common_lead_time` -- (``\\hat{l}``) common lead time for all production periods.
- `late_deviation` -- (``\\zeta^+_t``) maximal late lead time deviation for period `t`, t <- 1,...,T.
- `early_deviation` -- (``\\zeta^-_t``) maximal early lead time deviation for period `t`, t <- 1,...,T.
## Future period parameters
All data in future periods is assumed to be stored in a 1-indexed vector, where for some vector `v` describing future periods, `v[t]` is a description of period `T+t`.
- `demand` -- (``d_t``) demand in future period `T+t`, t <- 1,...,Tplus-T.
- `inventory_cost` -- (``c^I_t``) unit cost of storing product inventory from period `T+t` to period `T+t+1`, t <- 1,...,Tplus-T.
- `backordering_cost` -- (``c^B_t``) unit cost of backordering from period `T+t+1` to `T+t`, t <- 1,...,Tplus-T.
## Uncertainty modelling
- `uncertainty` -- (``\\mathcal{L}``) description of lead-time uncertainty set (see [`Uncertainty`](@ref)).
## R* bound
- `rbound` -- (optional) R* bound for the instance (``c``), provide nothing if not an R* instance.
"""
struct LotSizingInstance
    # time periods
    T :: Int64
    Tplus :: Int64
    # production period parameters
    setup_cost :: Vector{Float64}
    production_cost :: Vector{Float64}
    production_capacity :: Vector{Float64}
    # lead time characterization
    common_lead_time :: Int64
    late_deviation :: Vector{Int64}
    early_deviation :: Vector{Int64}
    # future periods
    demand :: Vector{Float64}
    inventory_cost :: Vector{Float64}
    backordering_cost :: Vector{Float64}
    # uncertainty modelling
    uncertainty :: Uncertainty.Model
    # R* bound
    rbound :: Union{Nothing, Float64}
end #LotSizingInstance

"""
Save LotSizingInstance to a file.

Arguments:
* inst - instance to be saved,
* path - path to file.

Optional arguments:
* name - instance name to add to header
* author - author name to add to header
"""
function save_instance(
    inst::LotSizingInstance,
    path::String
    ;
    name=nothing,
    author=nothing,
)
    write_array(f, a) = write(f, join(a, " ") * "\n")
    write_uncertainty(f, unc) = write(f, "$(Uncertainty.show(unc))\n")
    write_rbound(f, rbound) = write(f, isnothing(rbound) ? "NONE\n" : "R* $(rbound)\n")

    open(path, "w") do file
        if !isnothing(name)
            write(file, "# Instance: $(name)\n")
        end

        if !isnothing(name)
            write(file, "# Author: $(author)\n")
        end

        # time periods
        write(file, "# time periods\n")
        write(file, "$(inst.T) $(inst.Tplus)\n")

        # production periods
        write(file, "# production periods\n")
        write(file, "## setup cost\n")
        write_array(file, inst.setup_cost)
        write(file, "## production cost\n")
        write_array(file, inst.production_cost)
        write(file, "## production capacity\n")
        write_array(file, inst.production_capacity)

        # lead time characterization
        write(file, "# lead time\n")
        write(file, "## common lead time\n")
        write(file, "$(inst.common_lead_time)\n")
        write(file, "## late deviation\n")
        write_array(file, inst.late_deviation)
        write(file, "## early deviation\n")
        write_array(file, inst.early_deviation)

        # future periods
        write(file, "# future periods\n")
        write(file, "## demand\n")
        write_array(file, inst.demand)
        write(file, "## inventory cost\n")
        write_array(file, inst.inventory_cost)
        write(file, "## backordering cost\n")
        write_array(file, inst.backordering_cost)

        # uncertainty
        write(file, "# uncertainty\n")
        write_uncertainty(file, inst.uncertainty)

        # R* bound
        write(file, "# R* bound\n")
        write_rbound(file, inst.rbound)
    end

    return true
end

"Enumerate production periods (1,..,T) for the problem instance."
@memoize prod_periods(
    inst::LotSizingInstance,
)::UnitRange{Int64} = 1:inst.T

"Enumerate future periods (T+1,..,Tplus) for the problem instance."
@memoize future_periods(
    inst::LotSizingInstance,
)::UnitRange{Int64}  = inst.T+1:inst.Tplus

"Compute vector of maximal lead times for the problem instance."
@memoize function lead_max(inst::LotSizingInstance)::Vector{Int64}
    (;T,common_lead_time,late_deviation) = inst

    return [
        common_lead_time + late_deviation[i]
            for i in 1:T
    ]
end #lead_max

"Compute vector of minimal lead times for the problem instance"
@memoize function lead_min(inst::LotSizingInstance)::Vector{Int64}
    (;T,common_lead_time,early_deviation) = inst

    return [
        common_lead_time - early_deviation[i]
            for i in 1:T
    ]
end #lead_min

"Compute feasible lead times for production period `t`."
@memoize feasible_lead_times(
    inst::LotSizingInstance,
    t::Int64,
)::Vector{Int64} = (lead_min(inst)[t]):(lead_max(inst)[t])

"Compute feasible arrival times for production period `t`."
@memoize feasible_arrival_times(
    inst::LotSizingInstance,
    t::Int64,
)::Vector{Int64} = [t + l for l in feasible_lead_times(inst, t)]

"""
Compute vector of cumulative demands in future periods:
`cumulative_demand(inst)[t]` is the cumulative demand in period `T+t`  (``D_{T+t}``).
"""
@memoize cumulative_demand(
    inst::LotSizingInstance,
)::Vector{Int64} = cumsum(inst.demand)

function _must_arrive_by_t(
    inst::LotSizingInstance,
    t::Int64,
)::Int64
    (;T) = inst

    imax = 0
    for i in 1:T
        if i + lead_max(inst)[i] <= t
            imax = i
        end
    end

    return imax
end

"""
Compute vector of the numbers of deliveries that must arrive in future periods:
`must_arrive_by(inst)[t]` -> ``I_{T+t}`` (from section 4.1).
"""
@memoize function must_arrive_by(
    inst::LotSizingInstance,
)::Vector{Int64}
    (;Tplus) = inst

    return [
        _must_arrive_by_t(inst, t)
            for t in 1:Tplus
    ]
end

function _may_arrive_by_t(
    inst::LotSizingInstance,
    t::Int64,
)::Int64
    (;T) = inst

    imax = 0
    for i in 1:T
        if i + lead_min(inst)[i] <= t
            imax = i
        end
    end

    return imax
end

"""
Compute vector of the numbers of deliveries that may arrive in future periods:
`may_arrive_by(inst)[t]` -> ``J_{T+t}`` (from section 4.1).
"""
@memoize function may_arrive_by(
    inst::LotSizingInstance,
)::Vector{Int64}
    (;Tplus) = inst

    return [
        _may_arrive_by_t(inst, t)
            for t in 1:Tplus
    ]
end

"""
Check if the instance satisfies the assumptions made in the article.
"""
function validate_instance(
    inst::LotSizingInstance,
)::Vector{String}
    errors = []

    # TODO: Verify all the assumptions from the article are checked.

    # Check that all lead times are feasible
    for t in prod_periods(inst)
        if t + inst.common_lead_time + inst.late_deviation[t] > inst.Tplus
            push!(errors,
                "Production from t=$(t) may arrive after Tplus")
        end

        if t + inst.common_lead_time - inst.early_deviation[t] < inst.T + 1
            push!(errors,
                "Production from t=$(t) may arrive before T+1")
        end
    end

    return errors
end #validate_instance

"""
Compute the cost of production plan `x` (production quantity), `y` (production setup)
under arrival times vector `arrival`.

Arguments:
* inst - LotSizingInstance,
* x - vector of production quantity,
* y - vector of production setup,
* arrival - vector of arrival times.

Returns: cost of plan `(x,y)` under arrival times `arrival`
"""
function prod_lead_cost(inst::LotSizingInstance, x, y, arrival)
    (;T) = inst
    production = sum(inst.production_cost[t] * x[t] for t in prod_periods(inst))
    setup = sum(inst.setup_cost[t] * y[t] for t in prod_periods(inst))

    X = [sum(x[p] for p in prod_periods(inst) if arrival[p] <= t; init=0)
        for t in future_periods(inst)]

    D = cumulative_demand(inst)

    invt = [inst.inventory_cost[t-T]*max(0, X[t-T] - D[t-T])
        for t in future_periods(inst)]
    inventory = sum(invt)

    backt = [inst.backordering_cost[t-T]*max(0, D[t-T] - X[t-T])
        for t in future_periods(inst)]
    backordering = sum(backt)

    return production + setup + inventory + backordering
end #prod_lead_cost

function validate_gammaD(arr, common_lead, gamma)
    usage = sum(
        (p+common_lead != t) ? 1 : 0
            for (p,t) in enumerate(arr); init=0)

    return usage <= gamma
end #validate_gammaD

function validate_gammaC(arr, common_lead, gamma)
    usage = sum(
        abs(t - (p+common_lead))
            for (p,t) in enumerate(arr); init=0)

    return usage <= gamma
end #validate_gammaC

function valid_budget_usage(inst::LotSizingInstance, arr)
    (;type) = inst.uncertainty
    if type == Uncertainty.Interval
        return true
    elseif type == Uncertainty.GammaD
        return validate_gammaD(arr, inst.common_lead_time, inst.uncertainty.gamma)
    elseif type == Uncertainty.GammaC
        return validate_gammaC(arr, inst.common_lead_time, inst.uncertainty.gamma)
    end

    throw("compute_budget_usage: Unhandled uncertainty type: $(type)")
end #valid_budget_usage

function valid_arrival(inst, arr, ignore_budget::Bool)
    for (l,r) in zip(arr, arr[begin+1:end])
        if l > r return false end
    end

    return ignore_budget || valid_budget_usage(inst, arr)
end #valid_arrival


@memoize function list_scenarios(inst::LotSizingInstance;ignore_budget::Bool=false)
    l = inst.common_lead_time
    ranges = [(t+l-early):(t+l+late) for (early,late,t) in
        zip(inst.early_deviation, inst.late_deviation, 1:inst.T)]

    possible_arrivals = reshape(collect(Base.product(ranges...)), :)

    return [collect(arr) for arr in possible_arrivals if valid_arrival(inst, arr, ignore_budget)]
end #list_scenarios

"""
Create a vector of samples of arrival times based on an instance. Creates list of all arrivals to sample from

Arguments:
* inst - LotSizingInstance,
* N - number of sample scenarios.

Optional arguments:
* seed - (default: nothing) seed to RNG engine,
* ignore_budget - (default: nothing) sample the full uncertainty set ignoring the budget imposed in the instance,
* all_scenarios - (default: nothing) list of all scenarios to be used instead of computing them

Returns: vector of vectors of arrival times.
"""
function sample_arrivals(
    inst::LotSizingInstance,
    N::Int,
    ;
    seed=nothing,
    ignore_budget=false,
    all_scenarios=nothing,
)
    rng = Random.Xoshiro(seed)
    scenarios = !isnothing(all_scenarios) ? all_scenarios :
        Instance.list_scenarios(inst;ignore_budget)

    return [scenarios[Random.rand(rng, 1:length(scenarios))] for _ in 1:N]
end #sample_arrivals

"""
Compute maximal possible GammaC budget usage in an instance.
"""
@memoize function max_gammaC_budget(inst::LotSizingInstance)
    return sum(max(inst.early_deviation[t], inst.late_deviation[t])
        for t in 1:inst.T)
end

struct SamplingWeights
    # weight of previous
    weight::Int

    # weights for next choice
    # INV: weight = sum(n.weight for n in next) if length(prev) < inst.T
    next_weights::Vector{SamplingWeights}
end

Base.show(io::IO, w::SamplingWeights) = print(io,
    "($(w.weight), " *
    "$(Utils.print_array(w.next_weights)))"
)

@memoize function get_next_choices(inst::LotSizingInstance)
    choices = Vector{Int}[]
    sizehint!(choices, inst.T)
    for t in 1:inst.T
        early = t + inst.common_lead_time - inst.early_deviation[t]
        late = t + inst.common_lead_time + inst.late_deviation[t]

        push!(choices, collect(early : late))
    end

    return choices
end

function get_max_budget(
    inst::LotSizingInstance,
    ignore_budget::Bool
)
    if ignore_budget
        return 0
    end

    if inst.uncertainty.type in [Uncertainty.GammaD, Uncertainty.GammaC]
        return inst.uncertainty.gamma
    end

    return 0
end

function compute_sampling_weights(
    inst::LotSizingInstance,
    state::Array{Union{Missing,SamplingWeights}},
    choices::Vector{Vector{Int}},
    t::Int,
    prev::Union{Missing,Int}=missing,
    usage::Int=1,
)
    next = @views choices[t]

    weights = []
    sizehint!(weights, length(next))

    for t1 in next
        if !ismissing(prev) && prev > t1
            push!(weights, SamplingWeights(0, []))
            continue
        end

        nextusage = usage

        # TODO: support budget usage
        lookup = state[t, t1-inst.T, nextusage]

        if !ismissing(lookup)
            push!(weights, lookup)
            continue
        end

        t1weight = t == inst.T ?
            SamplingWeights(1, []) :
            compute_sampling_weights(inst, state, choices, t+1, t1, nextusage)
        state[t, t1-inst.T, usage] = t1weight
        push!(weights, t1weight)
    end

    weight = sum(w.weight for w in weights; init=0)

    return SamplingWeights(weight, weights)
end

function get_sampling_weights(
    inst::LotSizingInstance
    ;
    ignore_budget=false
)
    budget = get_max_budget(inst, ignore_budget)
    choices = get_next_choices(inst)
    # t,t1-T,1+usage
    state = Array{Union{Missing,SamplingWeights}}(missing,
        inst.T,
        inst.Tplus - inst.T,
        1 + budget,
    )

    return compute_sampling_weights(inst, state, choices, 1, missing, 1)
end

function weighted_sampling(
    inst::LotSizingInstance,
    N::Int,
    ;
    ignore_budget=false,
    seed=nothing,
)
    rng = Random.Xoshiro(seed)
    choices = get_next_choices(inst)
    weights = get_sampling_weights(inst; ignore_budget)

    scenarios = [zeros(Int, inst.T) for _ in 1:N]

    for i in 1:N
        w = weights
        r = Random.rand(rng, 1:w.weight)

        for t in 1:inst.T
            ch = choices[t]

            for (j, t1) in enumerate(ch)
                if r <= 0
                    throw("Error during generation of scenario $(i) in period $(t): exhausted possible scenarios.")
                end

                w1 = w.next_weights[j]

                if w1.weight > 0 && r <= w1.weight
                    scenarios[i][t] = t1
                    w = w1
                    break
                end

                r -= w1.weight
            end
        end
    end

    return scenarios
end #weighted_sampling

end #Instance
