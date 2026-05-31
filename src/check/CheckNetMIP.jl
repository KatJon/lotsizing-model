import LotsizingModel as LM

import .LM.Instance
import .LM.Model
import .LM.MipModel
import .LM.MinMaxModel
import .LM.MinMinModel

using JuMP: JuMP, optimize!, objective_value
using Accessors: @set,@reset

function compare_models(mnet, mmip)
    optimize!(mnet)
    optimize!(mmip)

    vnet = objective_value(mnet)
    vmip = objective_value(mmip)
    println("Network: $(vnet)")
    println("MIP: $(vmip)")

    if !isapprox(vmip, vnet)
        println("!!! Discrepancy in results")
    end
end

function eval_solution(
    inst,
    model, xs,
    neteval,
    mipeval,
)
    objval = objective_value(model)
    println("Model cost: $(objval)")

    netscen = neteval(inst, xs)
    optimize!(netscen)
    vnet = objective_value(netscen)

    if !isapprox(objval, vnet)
        println("!!! Model and net differ: $(vnet)")
    else
        println("Model and net align")
    end

    mipscen = mipeval(inst, xs)
    optimize!(mipscen)
    vmip = objective_value(mipscen)

    if !isapprox(objval, vmip)
        println("!!! Model and mip differ: $(vmip)")
    else
        println("Model and mip align")
    end
end

function compare_scenarios(
    inst::Instance.LotSizingInstance,
    mnet,xnet,
    mmip,xmip,
    neteval,
    mipeval
)
    println("## Network solution:")
    eval_solution(
        inst,
        mnet, xnet,
        neteval,
        mipeval,
    )

    println("## MIP solution:")
    eval_solution(
        inst,
        mmip, xmip,
        neteval,
        mipeval,
    )
end

function run(path="data/simpl.dat")
    inst = LM.load_instance(path)
    budget = Instance.Uncertainty.Model(Instance.Uncertainty.GammaD, 2)
    inst = @set inst.uncertainty=budget

    println("# Optimistic")
    moptnet,xoptnet,yoptnet = Model.create_minmin_model(inst)
    moptmip,xoptmip,yoptmip = MipModel.mip_minmin_model(inst)

    compare_models(moptnet, moptmip)

    compare_scenarios(inst,
        moptnet, collect(JuMP.value.(xoptnet)),
        moptmip, collect(JuMP.value.(xoptmip)),
        MinMinModel.opt_scenario_cost,
        MipModel.opt_scenario_cost,
    )

    #=================#

    println("# Pessimistic")
    mpesnet,xpesnet = Model.create_minmax_model(inst)
    mpesmip,xpesmip = MipModel.mip_minmax_model(inst)

    compare_models(mpesnet, mpesmip)
    compare_scenarios(inst,
        mpesnet, collect(JuMP.value.(xpesnet)),
        mpesmip, collect(JuMP.value.(xpesmip)),
        MinMaxModel.pes_scenario_cost,
        MipModel.pes_scenario_cost,
    )
    #=================#

    println("# R* model")
    pesval = objective_value(mpesnet)
    inst1 = @set inst.rbound=1.2*pesval
    mrnet,xrnet = Model.create_R_model(inst1)
    mrmip,xrmip = MipModel.mip_R_model(inst1)

    compare_models(mrnet, mrmip)
    compare_scenarios(inst,
        mrnet, collect(JuMP.value.(xrnet)),
        mrmip, collect(JuMP.value.(xrmip)),
        MinMinModel.opt_scenario_cost,
        MipModel.opt_scenario_cost,
    )
end
