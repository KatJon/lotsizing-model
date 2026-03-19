import LotsizingModel as LM
using .LM.Instance

import Plots as plt
using Accessors: @set,@reset
using ProgressMeter
using JuMP

function opt_value(inst::LotSizingInstance)
    model, = LM.Model.create_opt_model(inst)

    optimize!(model)

    return objective_value(model)
end

function pes_value(inst::LotSizingInstance)
    model, = LM.Model.create_pes_model(inst)

    optimize!(model)

    return objective_value(model)
end

function experimentR(
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

    opt_bound = opt_value(inst)
    pes_bound = pes_value(inst)

    RN = 64
    bounds=collect(LinRange(opt_bound, 1.5*pes_bound, RN))

    println("opt=$(opt_bound) pes=$(pes_bound)")

    results = zeros(length(bounds))
    solvetime = zeros(length(bounds))
    @showprogress for (i,rbound) in enumerate(bounds)
        inst1 = @set inst.rbound=rbound

        model, = LM.Model.create_R_model(inst1)

        optimize!(model)

        rtime = solve_time(model)
        if !is_solved_and_feasible(model)
            pmodel, = LM.Model.create_pes_model(inst1)
            optimize!(pmodel)

            if !is_solved_and_feasible(pmodel)
                println("Pes Model infeasible.")
                continue
            end

            results[i] = objective_value(pmodel)
            solvetime[i] = rtime + solve_time(pmodel)

            continue
        end

        results[i] = objective_value(model)
        solvetime[i] = solve_time(model)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, bounds, results; markershape=:circle,)
    plt.plot!(cost_plot, bounds, bounds)
    plt.vline!(cost_plot,[opt_bound,pes_bound])
    plt.title!(cost_plot, "R* cost")
    plt.gui(cost_plot)

    # println("Press any key to continue."); readline()

    # time_plot = plt.plot(rs, solvetime)
    # plt.scatter!(time_plot, rs, solvetime)
    # plt.title!(time_plot, "Solve time")
    # plt.gui(time_plot)

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig(cost_plot, "fig/$(name)_R.png")
    end
end

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
