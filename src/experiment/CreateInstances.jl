import LotsizingModel as LM

include("./GenInstance.jl")
using .GenInstance

"""
Generate an instance for LotSizing Problem

Arguments:
* T - number of production periods
* Tplus - number of all periods (production + future)

Keyword arguments:
* seed - (default: nothing) seed for RNG engine

Returns: LotSizingInstance
"""
function single_instance(
    T::Int, Tplus::Int
    ;
    seed=nothing
)
    MAX_DEMAND = 750
    CAPACITY = cld((Tplus-T)*MAX_DEMAND, T)

    params = InstanceParameters(
        ## production periods
        (10,50), #production_cost
        (1000,5000), #setup_cost
        (CAPACITY,fld(3*CAPACITY, 2)), #production_capacity
        ## future periods
        (cld(MAX_DEMAND,10),MAX_DEMAND), #demand
        (5,10), #inventory_cost
        (50,100), #backordering_cost
    )

    lt_params = UniformLeadTime(
        T, #common_lead_time
        (2, div(T,2)), #late_deviation
        (1, div(T,4)), #early_deviation
    )

    return gen_uniform(
        T,
        Tplus,
        params,
        lt_params,
        far_future_cost = 10^6,
        ;
        seed=seed
    )
end

function run(T, Tplus, path)
    inst = single_instance(T, Tplus)

    LM.Instance.save_instance(inst, path)
end
