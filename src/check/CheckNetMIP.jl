import LotsizingModel as LM

import .LM.Model
import .LM.MipModel

using JuMP: optimize!, objective_value
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

function run()
    inst = LM.load_instance("data/simpl.dat")
    
    println("Optimistic")
    moptnet, = Model.create_opt_model(inst)
    moptmip, = MipModel.opt_model(inst)

    compare_models(moptnet, moptmip)

    #=================#

    println("Pessimistic")
    mpesnet, = Model.create_pes_model(inst)
    mpesmip, = MipModel.pes_model(inst)

    compare_models(mpesnet, mpesmip)
    
    #=================#

    println("R* model")
    pesval = objective_value(mpesnet)
    inst1 = @set inst.rbound=1.2*pesval
    mrnet, = Model.create_R_model(inst)
    mrmip, = nothing, #TODO:

    optimize!(mrnet)

    # TODO: uncomment
    # compare_models(mrnet, mrmip)
end
