# Single-item lot sizing problem under budgeted lead-time uncertainty

This repository contains impementations of models, problem instances, and computational results accompanying the manuscript.

## Repository structure

* `lotsizing-model`
    - `data`
        + `gen`
    - `results`
    - `src`
        + `check`
        + `experiment`
        + `lib`

Directory `data` contains description of the format of the instance file (`format.dat`), two small instances used in manual testing (`simpl.dat` and `pesfig.dat`), and folder `gen` contains generated instances for the.

Directory `results` contains the computational results and figures presented in the article.

Directory `src` contains validation checks (folder `check`), experiments (folder `experiment`), and library of implementations of the models (folder `lib`).

## Obtaining the repository

The repository can be downloaded as a ZIP file from [GitHub](https://github.com/KatJon/lotsizing-model) or cloned with command:

```
git clone git@github.com:KatJon/lotsizing-model.git
```

## Installing the dependencies

To install dependencies in the development directory of the package the following commands can be used:

```
julia --project=.
] activate .
instantiate
```

## Adding package to environement

To add the package to be used in the other project the following command can be used:

```
] add "git@github.com:KatJon/lotsizing-model"
```

## Run the package

We provide a simple script `src/run.jl`, which loads the `LotsizingModel` package (imported as `LM`) provided in this repository and loads a simple instance. The environment can be ran interactively using command:

```
julia -i --project=. src/run.jl
```

## Run experiments

The experiments used to produce computational results presented in the article are implemented in script `src/experiment/RExperiment.jl` and provide a function `run` that loads the relevant instances and executes the experiments. The script can be ran using the following command:

```
julia -i --project=. src/experiment/RExperiment.jl
julia> run()
```

Additionally, experiments for other files can be executed using function `experiment`, which takes a short name of the instance for sake of naming the output files and path to the instance. The experiments can be further customized using keyword arguments (see documentation in `src/experiment/RExperiment.jl`).

By default, an open source solver HiGHS is used. If you wish to use Gurobi (e.g. to fully replicate the experiments conducted in the article), pass an optimizer factory as keyword argument `make_optimizer`. To suppress the Gurobi set the `OutputFlag` to `0`, as in function:
```julia
function create_gurobi()
    env = Gurobi.Env(Dict{String,Any}(
        "OutputFlag" => 0
    ))

    return Gurobi.Optimizer(env)
end
```

## License

The source code is licensed under MIT license. See the file LICENSE for details.

### Instances

Problem instances are distributed under the CC BY 4.0 (Creative Commons Attribution 4.0 International License).
Users are required to cite the associated manuscript when using these datasets.

## Citation

If you reference the article, use this code or dataset, please cite the manuscript as:

```
[Citation will be updated after publication]
```

## Contact

For questions regarding the article, implementation, or datasets, please contact:

Szymon Wróbel

Department of Fundamentals of Computer Science

Wrocław University of Science and Technology

email: [szymon.wrobel@pwr.edu.pl](mailto:szymon.wrobel@pwr.edu.pl)