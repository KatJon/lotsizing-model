import LotsizingModel as LM

using JuMP

using .LM.Instance
import .LM.Deterministic

using Accessors: @set

function get_cost((model,))
    optimize!(model)

    if !is_solved_and_feasible(model)
        throw("Cannot solve model")
    end

    return objective_value(model)
end

function checkGamma0(inst::LotSizingInstance)
    det_cost = get_cost(Deterministic.create_deterministic(inst))

    g1inst = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma1, 0)
    g2inst = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma2, 0)

    optg1_cost = get_cost(LM.Model.create_opt_model(g1inst))
    optg2_cost = get_cost(LM.Model.create_opt_model(g2inst))

    pesg1_cost = get_cost(LM.Model.create_pes_model(g1inst))
    pesg2_cost = get_cost(LM.Model.create_pes_model(g2inst))

    rg1inst = @set g1inst.rbound=pesg1_cost
    rg2inst = @set g2inst.rbound=pesg2_cost
    Rg1_cost = get_cost(LM.Model.create_R_model(rg1inst))
    Rg2_cost = get_cost(LM.Model.create_R_model(rg2inst))
    
    println("Deterministic:")
    println("$(det_cost)")

    println("Optimistic:")
    println("Gamma1: $(optg1_cost)")
    println("Gamma2: $(optg2_cost)")

    println("Pesimistic:")
    println("Gamma1: $(pesg1_cost)")
    println("Gamma2: $(pesg2_cost)")

    println("R*:")
    println("Gamma1: $(Rg1_cost)")
    println("Gamma2: $(Rg2_cost)")
end
