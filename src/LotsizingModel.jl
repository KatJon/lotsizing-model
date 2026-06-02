__precompile__(true)
module LotsizingModel

# Order of includes is significant
include("lib/Utils.jl")
include("lib/Instance.jl")
include("lib/Parse.jl")
include("lib/graph/MinMinGraph.jl")
include("lib/graph/MinMaxGraph.jl")
include("lib/graph/PseudoPolyGraph.jl")
include("lib/model/CommonModel.jl")
include("lib/model/MinMinModel.jl")
include("lib/model/MinMaxModel.jl")
include("lib/model/RModel.jl")
include("lib/model/Model.jl")
include("lib/model/PseudoPoly.jl")
include("lib/model/SAA.jl")
include("lib/model/Hurwicz.jl")
include("lib/model/Deterministic.jl")
include("lib/model/MipModel.jl")

export Utils, Instance, Parse
export MinMinGraph, MinMaxGraph, PseudoPolyGraph
export Model
export PseudoPoly
export SAA

import JuMP

export load_instance

"""
Load an instance for lot-sizing problem from file under `path`.

Arguments:
* path - path to file.

Returns: LotSizingInstance or nothing in case of validation error
"""
function load_instance(path::String)::Instance.LotSizingInstance
    open(path) do file
        instance = Parse.parse_instance(file)

        failed = run_validation(instance)
        if failed return nothing end

        return instance
    end
end

function run_validation(instance::Instance.LotSizingInstance)
    errors = Instance.validate_instance(instance)   

    if length(errors) > 0
        println("Validation of instance constraints failed")
        println("Encountered issues:")

        for error in errors
            println("* $(error)")
        end

        return true
    end

    return false
end

"Manual execution of Min-Max model with solution display."
function run_minmax_model(instance::Instance.LotSizingInstance)
    println("Building min-max model")
    @time model,x,y,z = Model.create_minmax_model(instance)
    
    JuMP.optimize!(model)

    if !JuMP.is_solved_and_feasible(model)
        println("MinMaxModel: Instance is infeasible")
        return
    end

    for t in Instance.prod_periods(instance)
        println("x[$(t)] => $(JuMP.value(x[t]))")
    end

    println("Optimal min-max objective: $(JuMP.objective_value(model))")
end #run_minmax_model

"Manual execution of Min-Min model with solution display."
function run_minmin_model(instance::Instance.LotSizingInstance)
    println("Building min-min model")
    @time model,x,_ = Model.create_minmin_model(instance)

    JuMP.optimize!(model)

    if !JuMP.is_solved_and_feasible(model)
        println("MinMinModel: Instance is infeasible")
        return
    end

    for t in Instance.prod_periods(instance)
        println("x[$(t)] => $(JuMP.value(x[t]))")
    end

    println("Min-min objective value: $(JuMP.objective_value(model))")
end #run_opt_model

"Manual execution of R* model with solution display."
function run_R_model(instance::Instance.LotSizingInstance)
    println("Building R* model")
    if isnothing(instance.rbound)
        println("Missing R* bound")
    end
    println("R* bound: $(instance.rbound)")
    @time model,x,y,rho,z,pgraph = Model.create_R_model(instance)

    JuMP.optimize!(model)

    if !JuMP.is_solved_and_feasible(model)
        println("R*Model: Instance is infeasible")
        return
    end

    for t in Instance.prod_periods(instance)
        println("x[$(t)] => $(JuMP.value(x[t]))")
    end
    
    println("Pessimistic objective:",
        JuMP.value(Model.minmax_objective(instance, pgraph, z,x,y)))
    println("R* objective value: $(JuMP.objective_value(model))")
end #run_R_model

"Manually run Min-Max, Min-Min, and R* models for instance stored in file `instance_path`."
function run_models(instance_path::String="../data/pesfig.dat")
    println("Parsing $(instance_path)")
    instance = load_instance(instance_path)
    if isnothing(instance)
        println("Instance didn't pass validation")
    end

    run_minmax_model(instance)

    println("\n===\n")

    run_minmin_model(instance)

    println("\n===\n")

    run_R_model(instance)
end #run_models

function __cold_start()
    instance = Instance.LotSizingInstance(
        # time periods
        1, 2,
        # prod params
        [0],
        [0],
        [0],
        # lead time
        1,
        [0],
        [0],
        # future
        [0],
        [0],
        [0],
        # uncertainty
        Instance.Uncertainty.Model(Instance.Uncertainty.GammaD, 1),
        100,
    )
    model1,_ = Model.create_minmax_model(instance)
    JuMP.optimize!(model1)
    model2,_ = Model.create_minmin_model(instance)
    JuMP.optimize!(model2)
    model3,_ = Model.create_R_model(instance)
    JuMP.optimize!(model3)
end

end #LotsizingModel
