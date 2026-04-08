import LotsizingModel as LM
using .LM.Instance

import .LM.StatUtils

import Plots as plt
using Plots.PlotMeasures
using Accessors: @set,@reset
using ProgressMeter
using JuMP
using DelimitedFiles

function hurwicz(
    inst::LotSizingInstance,
    ;
    name=nothing,
    Gamma1=nothing,
    Gamma2=nothing,
)
    if !isnothing(Gamma1)
        @reset inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma1, Gamma1)
    end
    
    if !isnothing(Gamma2)
        @reset inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma2, Gamma2)
    end
    alphas = collect(LinRange(0.0,1.0,50))
    
    results = zeros(length(alphas))
    solvetime = zeros(length(alphas))

    @showprogress for (i,alpha) in enumerate(alphas)
        model, = LM.Hurwicz.create_hurwicz_model(inst, alpha)

        optimize!(model)

        if !is_solved_and_feasible(model)
            println("Model cannot be solved for alpha=$(alpha), reason=$(termination_status(model))")
            continue
        end

        results[i] = objective_value(model)
        solvetime[i] = solve_time(model)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, alphas, results; markershape=:circle)
    plt.scatter!(cost_plot, alphas, results)
    plt.title!(cost_plot, "Hurwicz cost")

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig(cost_plot, "fig/$(name)_hurwicz.png")
    end
end

function run()
    
end
