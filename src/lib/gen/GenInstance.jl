module GenInstance

export
    InstanceParameters,
    LeadtimeParameters,
        UniformLeadTime,
    gen_uniform

import ..Instance
using ..Instance: Uncertainty

using Random

"""
Generate a single integer value sampled from range (from,to).

Arguments:
* rng - RNG engine
* (from,to) - tuple describing range (including endpoint)

Returns: integer value from range
"""
function generate_int(
    rng::Random.AbstractRNG,
    (from,to)::Tuple{Int,Int},
) :: Int
    return rand(rng, from:to)
end #generate_int

"""
Generate vector of random integer
"""
function generate_vector_int(
    rng::Random.AbstractRNG,
    N::Int,
    range::Tuple{Int,Int},
)
    return [generate_int(rng,range) for _ in 1:N]
end #generate_vector_int

"""
Structure describing the ranges of the non-leadtime parameters for instance generation.
"""
struct InstanceParameters
    # production periods
    production_cost :: Tuple{Int, Int}
    setup_cost :: Tuple{Int, Int}
    production_capacity :: Tuple{Int, Int}
    # future periods
    demand :: Tuple{Int,Int}
    inventory_cost :: Tuple{Int, Int}
    backordering_cost :: Tuple{Int, Int}
end #InstanceParameters

"Abstract type for structures describing lead time parameters"
abstract type LeadtimeParameters end

"""
Generate leadtime uncertainty set based on the supplied parameters.

Arguments:
* rng - RNG engine
* T - production horizon
* Tplus - future horizon
* params - parameters for leadtime generation

Returns: common lead time, vector of late deviation, vector of early deviation
"""
generate_leadtimes(
    rng::Random.AbstractRNG,
    T::Int,
    Tplus::Int,
    params::LeadtimeParameters,
) = throw("Unimplemented generate_leadtimes for $(typeof(params))")

"""
Structure describing uniform lead time structure (same early and late deviations, subject to horizon constraints).
"""
struct UniformLeadTime <: LeadtimeParameters
    common_lead_time :: Int
    late_deviation :: Tuple{Int, Int}
    early_deviation :: Tuple{Int, Int}
end # UniformLeadTime

"""
Specification of `generate_leadtimes` for `UniformLeadTime`.
"""
function generate_leadtimes(
    rng::Random.AbstractRNG,
    T::Int,
    Tplus::Int,
    params::UniformLeadTime,
)
    common_lead_time = params.common_lead_time
    late_deviation = zeros(Int, T)
    early_deviation = zeros(Int, T)

    common_late = generate_int(rng, params.late_deviation)
    common_early = generate_int(rng, params.early_deviation)

    for t in 1:T
        # t + common + late <= Tplus
        # late <= Tplus - t - common
        late_bound = Tplus - t - common_lead_time
        if late_bound < 0 throw("No feasible late deviation for t=$(t)") end
        late_deviation[t] = min(common_late, late_bound)
        
        # t + common - early >= T + 1
        # early <= t + common - T - 1
        early_bound = t+common_lead_time-T-1
        if early_bound < 0 throw("No feasible early deviation for t=$(t)") end
        early_deviation[t] = min(common_early, early_bound)
    end

    return common_lead_time, late_deviation, early_deviation
end #generate_leadtimes

# TODO: T=11, Tplus=21,41

"""
Generate an instance to LotSizing problem

Arguments:
* T - production horizon length
* Tplus - planning horizon length
* params - parameters for generating costs and demands
* leadtime_params - parameters for generating lead times

Optional arguments:
* seed - seed for RNG engine (default: nothing)
* far_future_cost - cost override for backordering from far future (default: nothing)
"""
function gen_uniform(
    T::Int,
    Tplus::Int,
    params::InstanceParameters,
    leadtime_params::LeadtimeParameters
    ;
    seed=nothing,
    far_future_cost=nothing
) :: Instance.LotSizingInstance
    rng = Xoshiro(seed)

    ## Production period parameters
    setup_cost = generate_vector_int(rng, T, params.setup_cost)

    production_cost = generate_vector_int(rng, T, params.production_cost)

    production_capacity = generate_vector_int(rng, T, params.production_capacity)

    ## Lead time characterization
    (common_lead_time, late_deviation, early_deviation) =
        generate_leadtimes(rng, T, Tplus, leadtime_params)

    ## Future period parameters
    demand = generate_vector_int(rng, Tplus-T, params.demand)

    inventory_cost = generate_vector_int(rng, Tplus-T, params.inventory_cost)

    backordering_cost = generate_vector_int(rng, Tplus-T, params.backordering_cost)

    # If `far_future_cost` is specified, override the cost for backordering from far future.
    if !isnothing(far_future_cost)
        backordering_cost[Tplus-T] = far_future_cost
    end

    # uncertainty
    uncertainty = Uncertainty.Model(Uncertainty.Interval, nothing)

    # R* bound
    rbound = nothing

    return Instance.LotSizingInstance(
        T, Tplus,
        float(setup_cost),
        float(production_cost),
        float(production_capacity),
        common_lead_time,
        late_deviation,
        early_deviation,
        float(demand),
        float(inventory_cost),
        float(backordering_cost),
        uncertainty,
        rbound,
    )
end # gen_uniform

end # GenInstance
