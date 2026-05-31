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

    gDinst = @set inst.uncertainty=Uncertainty.Model(Uncertainty.GammaD, 0)
    gCinst = @set inst.uncertainty=Uncertainty.Model(Uncertainty.GammaC, 0)

    optgD_cost = get_cost(LM.Model.create_opt_model(gDinst))
    optgC_cost = get_cost(LM.Model.create_opt_model(gCinst))

    pesgD_cost = get_cost(LM.Model.create_pes_model(gDinst))
    pesgC_cost = get_cost(LM.Model.create_pes_model(gCinst))

    rgDinst = @set gDinst.rbound=pesgD_cost
    rgCinst = @set gCinst.rbound=pesgC_cost
    RgD_cost = get_cost(LM.Model.create_R_model(rgDinst))
    RgC_cost = get_cost(LM.Model.create_R_model(rgCinst))
    
    println("Deterministic:")
    println("$(det_cost)")

    println("Optimistic:")
    println("GammaD: $(optgD_cost)")
    println("GammaC: $(optgC_cost)")

    println("Pesimistic:")
    println("GammaD: $(pesgD_cost)")
    println("GammaC: $(pesgC_cost)")

    println("R*:")
    println("GammaD: $(RgD_cost)")
    println("GammaC: $(RgC_cost)")
end
