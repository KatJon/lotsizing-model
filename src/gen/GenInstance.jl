module GenInstance

export gen_uniform

using Random

# TODO: T=11, Tplus=21,41

function gen_uniform(
    T::Int,
    Tplus::Int
    ;
    seed=nothing
)
    rng = isnothing(seed) ? Xoshiro() : Xoshiro(seed)



end # gen_uniform

end # GenInstance
