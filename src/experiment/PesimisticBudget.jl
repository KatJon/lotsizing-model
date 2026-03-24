import LotsizingModel as LM
using .LM.Instance
import .LM.StatUtils

import Plots as plt
using Accessors: @set,@reset
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
        plt.savefig(cost_plot,"fig/$(name)_cost_randG1.png")
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

function histG1(
    inst::LotSizingInstance
    ;
    N=100,
    name=nothing,
)
    println("Sampling test arrivals")
    samples = sample_arrivals(inst, N;ignore_budget=true)

    all_costs = []
    plots = []

    gammas = 0:inst.T

    @showprogress for gamma in gammas
        inst1 = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma1, gamma)
        model,x,y = LM.Model.create_pes_model(inst1)
        optimize!(model)

        xs = JuMP.value.(x)
        ys = JuMP.value.(y)
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

        push!(all_costs, costs)
    end

    amin = minimum(minimum(costs) for costs in all_costs)
    amax = maximum(maximum(costs) for costs in all_costs)

    # Sturges rule
    k = 1 + 2*ceil(Int64, log2(N))

    println("Preparing plots")
    @showprogress for (gamma,costs) in zip(gammas, all_costs)


        hist = StatUtils.mkhist(amin, amax, costs, k)
        hist = Float64.(hist)
        hist ./= Float64(N)

        bins = StatUtils.mkcutoffs(amin,amax, k)

        histplot = plt.bar(bins, hist, ylims=(0.0, 1.0))
        title = "\$\\Gamma_1 = $(gamma)\$"
        plt.title!(histplot,title)

        push!(plots, histplot)
    end

    allhist = plt.plot(plots...; layout=inst.T+1, size=(2000,1000))

    if isnothing(name)
        plt.gui(allhist)
    else
        plt.savefig("fig/$(name)_hist.png")
    end
end

function heatmapG1(
    inst::LotSizingInstance
    ;
    N=100,
    name=nothing,
    force=false,
)
    tmpname = isnothing(name) ? "TMP" : name
    SAVEFILE="output/$(tmpname)_heatmapG1_$(N).csv"
    

    gammas = 0:inst.T

    mkpath("output")
    all_costs = StatUtils.with_file_cache(SAVEFILE; force=force) do 
        samples = sample_arrivals(inst, N;ignore_budget=true)
        costs = zeros(Float64, length(gammas), length(samples))

        @showprogress for (i,gamma) in enumerate(gammas)
            inst1 = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma1, gamma)
            model,x,y = LM.Model.create_pes_model(inst1)
            optimize!(model)

            xs = JuMP.value.(x)
            ys = JuMP.value.(y)

            for (j, sample) in enumerate(samples)
                costs[i,j] = prod_lead_cost(inst,xs,ys,sample)
            end
        end

        return costs
    end

    # Sturges rule
    buckets = 1 + 2*ceil(Int64, log2(N))

    vmin = minimum(all_costs[i] for i in eachindex(all_costs))
    vmax = maximum(all_costs[i] for i in eachindex(all_costs))
    cutoffs = StatUtils.mkcutoffs(vmin, vmax, buckets)
    hist = zeros(Float64, length(gammas), buckets)

    for (i,_) in enumerate(gammas)
        hist[i,:] = StatUtils.mkhist(vmin, vmax, all_costs[i,:], buckets)
    end

    hist ./= N

    cont = plt.contourf(cutoffs[begin:end-1], gammas, hist, levels=20, color=:turbo, lw=0)
    plt.title!(cont, "Simulated cost (minmax model)")
    plt.xlabel!(cont, "cost")
    plt.ylabel!(cont, "\$\\Gamma_1\$")

    if isnothing(name)
        plt.gui(cont)
    else
        plt.savefig("fig/$(name)_pes_histcont.png")
    end
end


function run()
    inst = LM.load_instance("data/gen/10_21_1.dat")

    experimentG1(inst;name="10_21_1")
    experimentG2(inst;name="10_21_1")

    random_tests_G1(inst;name="10_21_1",N=500)
    random_tests_G2(inst;name="10_21_1",N=500)
end
