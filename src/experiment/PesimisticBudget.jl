import LotsizingModel as LM
using .LM.Instance

import Plots as plt
using Accessors: @set
using ProgressMeter
using JuMP
using Statistics

function experimentG1(inst::LotSizingInstance;name=nothing)
    budgets = 0:inst.T
    results = zeros(length(budgets))
    solvetime = zeros(length(budgets))

    @showprogress for (i,gamma) in enumerate(budgets)
        budget = Uncertainty.Model(Uncertainty.Gamma1, gamma)
        inst1 = @set inst.uncertainty=budget

        model, = LM.Model.create_pes_model(inst1)

        optimize!(model)

        if !is_solved_and_feasible(model)
            println("Model cannot be solved for Gamma=$(gamma), reason=$(termination_status(model))")
            continue
        end

        results[i] = objective_value(model)
        solvetime[i] = solve_time(model)
    end

    cost_plot = plt.plot(budgets, results)
    plt.scatter!(cost_plot, budgets, results)
    plt.title!(cost_plot, "Worst-case cost")
    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig("fig/$(name)_cost_pesG1.png")
    end

    # println("Press any key to continue."); readline()

    # time_plot = plt.plot(budgets, solvetime)
    # plt.scatter!(time_plot, budgets, solvetime)
    # plt.title!(time_plot, "Solve time")
    # plt.gui(time_plot)

end

function experimentG2(inst::LotSizingInstance;name=nothing)
    budgets = 0:fld(inst.T*inst.T, 2)
    results = zeros(length(budgets))
    solvetime = zeros(length(budgets))

    @showprogress for (i,gamma) in enumerate(budgets)
        budget = Uncertainty.Model(Uncertainty.Gamma2, gamma)

        inst1 = @set inst.uncertainty=budget

        model, = LM.Model.create_pes_model(inst1)

        optimize!(model)

        if !is_solved_and_feasible(model)
            println("Model cannot be solved for Gamma=$(gamma), reason=$(termination_status(model))")
            continue
        end

        results[i] = objective_value(model)
        solvetime[i] = solve_time(model)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, budgets, results;markershape=:circle,label="pes")
    plt.title!(cost_plot, "Worst-case cost")
    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig("fig/$(name)_cost_pesG2.png")
    end


    # time_plot = plt.plot(budgets, solvetime)
    # plt.scatter!(time_plot, budgets, solvetime)
    # plt.title!(time_plot, "Solve time")
    # plt.gui(time_plot)

end

function random_tests_G1(inst::LotSizingInstance;N=100,name=nothing)
    budgets = 0:inst.T
    pes_min = zeros(length(budgets))
    pes_avg = zeros(length(budgets))
    pes_max = zeros(length(budgets))

    println("Sampling test arrivals")
    samples = sample_arrivals(inst, N;ignore_budget=true)

    @showprogress for (i,gamma) in enumerate(budgets)
        budget = Uncertainty.Model(Uncertainty.Gamma1, gamma)
        inst1 = @set inst.uncertainty=budget

        model,x,y = LM.Model.create_pes_model(inst1)
        optimize!(model)

        xs = JuMP.value.(x)
        ys = JuMP.value.(y)
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

        pes_min[i] = minimum(costs)
        pes_avg[i] = mean(costs)
        pes_max[i] = maximum(costs)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, budgets, pes_min;markershape=:circle,label="min")
    plt.plot!(cost_plot, budgets, pes_avg;markershape=:circle,label="avg")
    plt.plot!(cost_plot, budgets, pes_max;markershape=:circle,label="max")
    plt.title!(cost_plot, "Min-max sampled real cost \$\\Gamma1\$")
    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig("fig/$(name)_cost_randG1.png")
    end
end

function random_tests_G2(inst::LotSizingInstance;N=100,name=nothing)
    budgets = 0:fld(inst.T*inst.T,2)
    pes_min = zeros(length(budgets))
    pes_avg = zeros(length(budgets))
    pes_max = zeros(length(budgets))

    println("Sampling test arrivals")
    samples = sample_arrivals(inst, N;ignore_budget=true)

    @showprogress for (i,gamma) in enumerate(budgets)
        budget = Uncertainty.Model(Uncertainty.Gamma2, gamma)
        inst1 = @set inst.uncertainty=budget

        model,x,y = LM.Model.create_pes_model(inst1)
        optimize!(model)

        xs = JuMP.value.(x)
        ys = JuMP.value.(y)
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

        pes_min[i] = minimum(costs)
        pes_avg[i] = mean(costs)
        pes_max[i] = maximum(costs)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, budgets, pes_min;markershape=:circle,label="min")
    plt.plot!(cost_plot, budgets, pes_avg;markershape=:circle,label="avg")
    plt.plot!(cost_plot, budgets, pes_max;markershape=:circle,label="max")
    plt.title!(cost_plot, "Min-max sampled real cost \$\\Gamma2\$")
    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig("fig/$(name)_cost_randG2.png")
    end
end
