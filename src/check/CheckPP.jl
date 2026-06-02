import LotsizingModel as LM

import .LM.Instance
import .Instance.Uncertainty
import .LM.MinMinModel
import .LM.PseudoPoly

import Gurobi

using JuMP
using Accessors: @set,@reset

function create_gurobi()
    env = Gurobi.Env(Dict{String,Any}(
        "OutputFlag" => 0
    ))

    return Gurobi.Optimizer(env)
end #create_gurobi


function run(path="data/pesfig.dat")
    LM.Utils.set_default_optimizer!(create_gurobi)

    inst = LM.load_instance(path)
    budget = Uncertainty.Model(Uncertainty.GammaD, 2)
    @reset inst.uncertainty=budget

    println("Solving MinMin")
    netm,netx,nety = MinMinModel.create_minmin_model(inst)
    optimize!(netm)
    netval = objective_value(netm)
    println("Solving PseudoPoly")
    ppval,ppx,ppy = PseudoPoly.solve_pseudopoly(inst)

    println("MinMin obj: $(netval)")
    println("PseudoPoly obj: $(ppval)")

    if !isapprox(netval, ppval)
        println("!!! Objective values diverge")
        return
    end

    mnetsc = MinMinModel.opt_scenario_cost(inst, collect(value.(netx)))
    optimize!(mnetsc)
    netsc = objective_value(mnetsc)
    mppsc = MinMinModel.opt_scenario_cost(inst, collect(ppx))
    optimize!(mppsc)
    ppsc = objective_value(mppsc)

    println("MinMin eval: $(netsc)")
    println("PseudoPoly eval: $(ppsc)")

    if !isapprox(netval, netsc)
        println("!!! MinMin objective and evaluation diverge")
    end

    if !isapprox(ppval, ppsc)
        println("!!! Pseudopoly objective and evaluation diverge")
    end
end