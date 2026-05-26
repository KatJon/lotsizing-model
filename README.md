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

```
julia --project=.
] activate .
instantiate
```

## Run the model

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

For questions regarding