__precompile__(true)
module LotsizingModel

# Order of includes is significant
include("lib/Utils.jl")
include("lib/Instance.jl")
include("lib/Parse.jl")
include("lib/graph/OptGraph.jl")
include("lib/graph/PesGraph.jl")
include("lib/model/Model.jl")
include("lib/model/PseudoPoly.jl")
include("lib/model/SAA.jl")
include("lib/model/Hurwicz.jl")
include("lib/model/Deterministic.jl")
include("lib/model/MipModel.jl")
include("lib/gen/GenInstance.jl")

export Utils, Instance, Parse
export OptGraph, PesGraph
export Model
export PseudoPoly
export SAA
export GenInstance

import JuMP

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

function run_pes_model(instance::Instance.LotSizingInstance;compute_optobj=false)
    println("Building pes model")
    @time model,x,y,z = Model.create_pes_model(instance)
    
    JuMP.optimize!(model)

    if !JuMP.is_solved_and_feasible(model)
        println("PesModel: Instance is infeasible")
        return
    end

    for t in Instance.prod_periods(instance)
        println("x[$(t)] => $(JuMP.value(x[t]))")
    end

    println("Optimal pessimistic objective: $(JuMP.objective_value(model))")
end #run_pes_model

function run_opt_model(instance::Instance.LotSizingInstance; compute_pesobj=false)
    println("Building opt model")
    @time model,x,rho,far_future_backordering = Model.create_opt_model(instance)

    JuMP.optimize!(model)

    if !JuMP.is_solved_and_feasible(model)
        println("OptModel: Instance is infeasible")
        return
    end

    for t in Instance.prod_periods(instance)
        println("x[$(t)] => $(JuMP.value(x[t]))")
    end

    println("Far future backordering: $(JuMP.value(far_future_backordering))")

    println("Optimistic objective value: $(JuMP.objective_value(model))")

    println("Optimistic arrivals scenario:")
    println(Model.get_opt_arrivals(instance, rho))
end #run_opt_model

function run_R_model(instance::Instance.LotSizingInstance)
    println("Building R* model")
    if isnothing(instance.rbound)
        println("Missing R* bound")
    end
    println("R* bound: $(instance.rbound)")
    @time model,x,rho,z,y,pgraph = Model.create_R_model(instance)

    JuMP.optimize!(model)

    if !JuMP.is_solved_and_feasible(model)
        println("R*Model: Instance is infeasible")
        return
    end

    for t in Instance.prod_periods(instance)
        println("x[$(t)] => $(JuMP.value(x[t]))")
    end
    
    println("Pessimistic objective:",
        JuMP.value(Model.pes_objective(instance, pgraph, z,x,y)))
    println("R* objective value: $(JuMP.objective_value(model))")
    println("R* arrivals scenario:")
    println(Model.get_opt_arrivals(instance, rho))
end #run_R_model

function run_models(instance_path::String="../data/pesfig.dat")
    println("Parsing $(instance_path)")
    instance = load_instance(instance_path)
    if isnothing(instance)
        println("Instance didn't pass validation")
    end

    run_pes_model(instance, compute_optobj=true)

    println("\n===\n")

    run_opt_model(instance, compute_pesobj=true)

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
        Instance.Uncertainty.Model(Instance.Uncertainty.Gamma1, 1),
        100,
    )
    model1,_ = Model.create_pes_model(instance)
    JuMP.optimize!(model1)
    model2,_ = Model.create_opt_model(instance)
    JuMP.optimize!(model2)
    model3,_ = Model.create_R_model(instance)
    JuMP.optimize!(model3)
end

end #LotsizingModel
