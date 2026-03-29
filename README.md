## Install dependencies

```
] activate .
instantiate
```

## Run the model

```
julia -i --project=. src/LotsizingModel.jl
```

## Run experiments

### Robust budgets (minmax, minmin)
```
julia -i --project=. src/experiment/RobustBudget.jl
julia> run()
```

### R* budget
```
julia -i --project=. src/experiment/RBudget.jl
julia> run()
```
