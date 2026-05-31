# Instances for Single-item Capacitated Lot-sizing Problem

File `format.dat` contains an annotated pseudo-instance describing format of the data files containing instances for the problem.

Files `simpl.dat` and `pesfig.dat` are simple instances used for manual testing and verification.

Folder `gen` contains generated instances used in the paper. The naming convention is `T_T+_i.dat`, where `T` is number of production periods, `T+` is number of all periods, and `i` is the number of instance.

## Data generation

Further instances may be generated using `src/experiment/CreateInstances.jl` script, for example:

```
julia -i --project=. src/experiment/CreateInstances.jl
julia> run(10,21,"data/gen/10_21_21.dat")
```