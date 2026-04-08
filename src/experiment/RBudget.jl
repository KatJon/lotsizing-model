import LotsizingModel as LM

import .LM.StatUtils
using .LM.Instance
using .LM.ExperimentUtils

import Plots as plt
using Plots.PlotMeasures
using Accessors: @set,@reset
using ProgressMeter
using JuMP
using DelimitedFiles
using Printf

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

function solveR(inst; Gamma1=nothing, Gamma2=nothing, rbound=nothing)
    if !isnothing(Gamma1)
        @reset inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma1, Gamma1)
    end
    
    if !isnothing(Gamma2)
        @reset inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma2, Gamma2)
    end

    if !isnothing(rbound)
        @reset inst.rbound=rbound
    end

    model,x,y = LM.Model.create_R_model(inst)

    optimize!(model)

    if !is_solved_and_feasible(model)
        pmodel,px,py = LM.Model.create_pes_model(inst)
        optimize!(pmodel)

        if !is_solved_and_feasible(pmodel)
            println("Pes Model infeasible.")
            return
        end

        return objective_value(pmodel),px,py
    end

    return objective_value(model),x,y
end

function experimentR(
    inst::LotSizingInstance,
    ;
    name=nothing,
    Gamma1=nothing,
    Gamma2=nothing,
)
    titlesuffix = ""
    if !isnothing(Gamma1)
        @reset inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma1, Gamma1)
        titlesuffix = "\$\\Gamma_1=$(Gamma1)"
    end
    
    if !isnothing(Gamma2)
        @reset inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma2, Gamma2)
        titlesuffix = "\$\\Gamma_2=$(Gamma2)"
    end

    opt_bound = opt_value(inst)
    pes_bound = pes_value(inst)

    RN = 64
    bounds=collect(LinRange(opt_bound, 1.5*pes_bound, RN))

    results = zeros(length(bounds))
    @showprogress for (i,rbound) in enumerate(bounds)
        inst1 = @set inst.rbound=rbound

        model, = LM.Model.create_R_model(inst1)

        optimize!(model)

        if !is_solved_and_feasible(model)
            pmodel, = LM.Model.create_pes_model(inst1)
            optimize!(pmodel)

            if !is_solved_and_feasible(pmodel)
                println("Pes Model infeasible.")
                continue
            end

            results[i] = objective_value(pmodel)

            continue
        end

        results[i] = objective_value(model)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, bounds, results; markershape=:circle,)
    plt.plot!(cost_plot, bounds, bounds)
    plt.vline!(cost_plot,[opt_bound,pes_bound])
    plt.title!(cost_plot, "R* cost \$\\Gamma=$(gamma1)")
    plt.gui(cost_plot)

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig(cost_plot, "fig/$(name)_R.png")
    end
end

function cost_RG1(
    inst::LotSizingInstance,
    ;
    name=nothing,
)
    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval)
    
    gammas = 0:inst.T
    rbounds = LinRange(pes_bound, 1.5*pes_bound, 5)
    plots = []

    for (i,rbound) in enumerate(rbounds)
        println("R* bound $(i)")
        inst1 = @set inst.rbound=rbound

        results = zeros(length(gammas))

        case_experiment(inst1, gammas,
            with_gamma1,
            LM.Model.create_R_model
        ) do i,model,_,_
            results[i] = objective_value(model)
        end

        cost_plot = make_cost_plot(gammas, results;
            title="R* cost \$\\Gamma_1\$ (b=$(@sprintf "%.2e" rbound))",
            ylabel="cost",
            xlabel="\$\\Gamma\$",
        )

        push!(plots, cost_plot)
    end

    allplots = plt.plot(plots...; layout=length(rbounds), size=(2000,1000))

    if isnothing(name)
        println("Can't show multiple plots interactively.")
    else
        plt.savefig(allplots, "fig/$(name)_RG1_cost.png")
    end
end

function sample_RG1(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
)
    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval)

    gammas = 0:inst.T
    rbounds = LinRange(pes_bound, 1.5*pes_bound, 5)
    samples = sample_arrivals(inst, N)

    plots=[]

    for (i,rbound) in enumerate(rbounds)
        println("R* bound $(i)")
        inst1 = @set inst.rbound=rbound

        pes_min = zeros(length(gammas))
        pes_max = zeros(length(gammas))

        case_experiment(inst1, gammas,
            with_gamma1,
            LM.Model.create_R_model
        ) do i,_,xs,ys
            costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

            pes_min[i] = minimum(costs)
            pes_max[i] = maximum(costs)
        end

        cost_plot = plt.Plot()
        plt.plot!(cost_plot, gammas, pes_min;
            markershape=:circle, label="min")
        plt.plot!(cost_plot, gammas, pes_max;
            markershape=:circle,label="max")
        plt.title!(cost_plot, "R* sampled cost \$\\Gamma_1\$ (b=$(@sprintf "%.2e" rbound))")
        plt.ylabel!(cost_plot, "cost")
        plt.xlabel!(cost_plot, "\$\\Gamma\$")

        push!(plots, cost_plot)
    end

    allplots = plt.plot(plots...; layout=length(rbounds), size=(2000,1000))

    if isnothing(name)
        println("Can't show multiple plots interactively.")
    else
        plt.savefig(allplots, "fig/$(name)_RG1_sample.png")
    end
end

function hist_RG1(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
)
    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval)

    gammas = 0:inst.T
    rbounds = LinRange(pes_bound, 1.5*pes_bound, 5)
    samples = sample_arrivals(inst, N)

    for (i,rbound) in enumerate(rbounds)
        println("R* bound $(i)")
        inst1 = @set inst.rbound=rbound

        all_costs = []
        case_experiment(inst1, gammas,
            with_gamma1,
            LM.Model.create_R_model
        ) do _,_,xs,ys
            costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

            push!(all_costs, costs)
        end

        amin = minimum(minimum(costs) for costs in all_costs)
        amax = maximum(maximum(costs) for costs in all_costs)

        # Sturges rule
        k = 1 + 2*ceil(Int64, log2(N))

        plots = plot_histograms(
            amin, amax, k, N,
            gammas,
            all_costs;
            mktitle=gamma -> "\$\\Gamma_1 = $(gamma)\$",
        )

        for plot in plots
            plt.vline!(plot, [rbounds[i]], legend=false)
        end

        allhist = plt.plot(plots...; layout=length(gammas), size=(2000,1000))

        if isnothing(name)
            println("Can't show multiple plots interactively.")
        else
            plt.savefig(allhist, "fig/$(name)_RG1_hist$(i).png")
        end
    end
end

function experiments(name,path)
    println("# Experiment \"$(name)\"")
    inst = LM.load_instance(path)

    N = 10000

    println("### Cost Gamma1")
        cost_RG1(inst; name=name)

    println("### Sample Gamma1")
        sample_RG1(inst; name=name, N=N)

    println("### Histogram Gamma1")
        hist_RG1(inst; name=name, N=N)
end

function run()
    mkpath("fig")

    experiments("10_21_1", "data/gen/10_21_1.dat")
end #run
