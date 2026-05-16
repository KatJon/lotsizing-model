import LotsizingModel as LM

import .LM.StatUtils
import .LM.RModel
import .LM.PesModel
import .LM.OptModel
using .LM.Instance
using .LM.ExperimentUtils

import Plots as plt
using Plots.PlotMeasures
using Accessors: @set,@reset
using DelimitedFiles
using JuMP
using Printf
using ProgressMeter
using StatsBase


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
    NBOUNDS = 6
    rbounds = LinRange(pes_bound, 1.5*pes_bound, NBOUNDS)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, NBOUNDS)]
    plots = []

    for (i,rbound) in enumerate(rbounds)
        println("R* bound $(i)")
        inst1 = @set inst.rbound=rbound

        results = zeros(length(gammas))

        case_experiment(inst1, gammas,
            with_gamma1,
            LM.Model.create_R_model;
            optimize=RModel.optimizeRmodel!,
        ) do i,model,_,_
            results[i] = objective_value(model)
        end

        cost_plot = make_cost_plot(gammas, results;
            title="R* cost \$\\Gamma_1\$ (b=$(rstring[i])pes)",
            ylabel="cost",
            xlabel="\$\\Gamma\$",
            bottom_margin=10mm,
        )

        push!(plots, cost_plot)
    end

    minval = minimum(plt.ylims(plot)[begin] for plot in plots)
    maxval = maximum(plt.ylims(plot)[end] for plot in plots)

    for plot in plots
        plt.ylims!(plot, minval, maxval)
    end

    allplots = plt.plot(plots...; layout=length(rbounds), size=(2000,1000))

    if isnothing(name)
        println("Can't show multiple plots interactively.")
    else
        plt.savefig(allplots, "fig/$(name)_RG1_cost.png")
    end
end

function cost_RG2(
    inst::LotSizingInstance,
    ;
    name=nothing,
)
    inst_interval = ExperimentUtils.with_interval(inst)
    pes_bound = pes_value(inst_interval)
    
    gammas = 0:max_gamma2_budget(inst)
    NBOUNDS = 6
    rbounds = LinRange(pes_bound, 1.5*pes_bound, NBOUNDS)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, NBOUNDS)]
    plots = []

    for (i,rbound) in enumerate(rbounds)
        println("R* bound $(i)")
        inst1 = @set inst.rbound=rbound

        results = zeros(length(gammas))

        case_experiment(inst1, gammas,
            with_gamma2,
            LM.Model.create_R_model;
            optimize=RModel.optimizeRmodel!
        ) do j,model,_,_
            results[j] = objective_value(model)
        end

        cost_plot = make_cost_plot(gammas, results;
            title="R* cost \$\\Gamma_2\$ (b=$(rstring[i])pes",
            ylabel="cost",
            xlabel="\$\\Gamma\$",
            bottom_margin=10mm,
        )

        push!(plots, cost_plot)
    end

    minval = minimum(plt.ylims(plot)[begin] for plot in plots)
    maxval = maximum(plt.ylims(plot)[end] for plot in plots)

    for plot in plots
        plt.ylims!(plot, minval, maxval)
    end

    allplots = plt.plot(plots...; layout=length(rbounds), size=(2000,1000))

    if isnothing(name)
        println("Can't show multiple plots interactively.")
    else
        plt.savefig(allplots, "fig/$(name)_RG2_cost.png")
    end
end #cost_RG2

function sample_RG1(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
)
    inst_interval = ExperimentUtils.with_interval(inst)
    pes_bound = pes_value(inst_interval)

    gammas = 0:inst.T
    NBOUNDS = 6
    rbounds = LinRange(pes_bound, 1.5*pes_bound, NBOUNDS)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, NBOUNDS)]

    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    plots=[]

    for (i,rbound) in enumerate(rbounds)
        println("R* bound $(i)")
        inst1 = @set inst.rbound=rbound

        pes_min = zeros(length(gammas))
        pes_max = zeros(length(gammas))
        p90 = zeros(length(gammas))
        
        stats = []

        case_experiment(inst1, gammas,
            with_gamma1,
            LM.Model.create_R_model;
            optimize=RModel.optimizeRmodel!
        ) do j,_,xs,ys
            costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

            expstats = ExperimentUtils.experiment_stats(costs)
            push!(stats, [gammas[j], expstats...])

            pes_min[j] = minimum(costs)
            pes_max[j] = maximum(costs)
            p90[j] = expstats.perc90
        end

        if !isnothing(name)
            ExperimentUtils.save_stats(
                "fig/$(name)_RG1_sample_stats_$(i).csv",
                stats,
            )
        end

        cost_plot = plt.Plot()
        plt.plot!(cost_plot, gammas, pes_min;
            markershape=:circle, label="min")
        plt.plot!(cost_plot, gammas, pes_max;
            markershape=:circle,label="max")
        plt.plot!(cost_plot, gammas, p90;
            label="90th perc.")
        plt.title!(cost_plot, "R* sampled cost \$\\Gamma_1\$ (b=$(rstring[i])pes)")
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
end #sample_RG1

function sample_override_RG1(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
)
    inst_interval = ExperimentUtils.with_interval(inst)
    pes_bound = pes_value(inst_interval)

    gammas = 0:inst.T
    rbounds = LinRange(pes_bound, 1.5*pes_bound, 6)
    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)
    plots=[]

    inst = with_gamma1(inst, 0)

    for (i,rbound) in enumerate(rbounds)
        inst1 = ExperimentUtils.with_rbound(inst,rbound)
        println("R* bound $(i)")
        pes_min = zeros(length(gammas))
        pes_max = zeros(length(gammas))

        stats = []
        
        @showprogress for (j,gamma) in enumerate(gammas)
            model,x,y = RModel.create_R_model(inst1;
                override_R_uncertainty=Uncertainty.Model(Uncertainty.Gamma1, gamma))
            
            model,x,y = RModel.optimizeRmodel!(inst,model,x,y)

            if !JuMP.is_solved_and_feasible(model)
                error("Model ($(i),$(case_param)) cannot be solved, reason=$(JuMP.termination_status(model))")
                continue
            end

            xs = collect(JuMP.value.(x))
            ys = collect(JuMP.value.(y))

            costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

            push!(stats, [gammas[j], ExperimentUtils.experiment_stats(costs)...])

            pes_min[j] = minimum(costs)
            pes_max[j] = maximum(costs)
        end

        if !isnothing(name)
            ExperimentUtils.save_stats(
                "fig/$(name)_RG1_sample_override_stats_$(i).csv",
                stats,
            )
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
        plt.savefig(allplots, "fig/$(name)_RG1_sample_override.png")
    end
end #sample_override_RG1

function sample_RG2(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
)
    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval)

    gammas = 0:max_gamma2_budget(inst)
    rbounds = LinRange(pes_bound, 1.5*pes_bound, 6)
    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    plots=[]

    for (i,rbound) in enumerate(rbounds)
        println("R* bound $(i)")
        inst1 = @set inst.rbound=rbound

        pes_min = zeros(length(gammas))
        pes_max = zeros(length(gammas))
        p90 = zeros(length(gammas))

        stats = []

        case_experiment(inst1, gammas,
            with_gamma2,
            LM.Model.create_R_model;
            optimize=RModel.optimizeRmodel!
        ) do j,_,xs,ys
            costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

            expstats = ExperimentUtils.experiment_stats(costs)
            push!(stats, [gammas[j], expstats...])

            pes_min[j] = minimum(costs)
            pes_max[j] = maximum(costs)
            p90[j] = expstats.perc90
        end

        if !isnothing(name)
            ExperimentUtils.save_stats(
                "fig/$(name)_RG1_sample_stats_$(i).csv",
                stats,
            )
        end

        cost_plot = plt.Plot()
        plt.plot!(cost_plot, gammas, pes_min;
            markershape=:circle, label="min")
        plt.plot!(cost_plot, gammas, pes_max;
            markershape=:circle,label="max")
        plt.plot!(cost_plot, gammas, p90;
            label="90th percentile")
        plt.title!(cost_plot, "R* sampled cost \$\\Gamma_2\$ (b=$(@sprintf "%.2e" rbound))")
        plt.ylabel!(cost_plot, "cost")
        plt.xlabel!(cost_plot, "\$\\Gamma\$")

        push!(plots, cost_plot)
    end

    allplots = plt.plot(plots...; layout=length(rbounds), size=(2000,1000))

    if isnothing(name)
        println("Can't show multiple plots interactively.")
    else
        plt.savefig(allplots, "fig/$(name)_RG2_sample.png")
    end
end #sample_RG2

function hist_RG1(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
)
    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval)

    gammas = 0:inst.T
    NBOUNDS=6
    rbounds = LinRange(pes_bound, 1.5*pes_bound, NBOUNDS)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, NBOUNDS)]
    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    maxval = 0
    all_plots = []

    for (i,rbound) in enumerate(rbounds)
        println("R* bound $(i)")
        inst1 = @set inst.rbound=rbound

        all_costs = []
        case_experiment(inst1, gammas,
            with_gamma1,
            LM.Model.create_R_model;
            optimize=RModel.optimizeRmodel!
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
            rescale=true
        )

        maxval = max(maxval, maximum(plt.ylims(plot)[end] for plot in plots))

        for plot in plots
            plt.vline!(plot, [rbounds[i]], legend=false)
        end

        push!(all_plots, plots)
    end

    for (i,rs) in enumerate(rstring)
        for plot in all_plots[i]
            plt.ylims!(plot, (0.0, maxval))
        end
        
        layout=@plt.layout [a{0.01h}; plt.grid(4,3)]

        allhist = plt.plot(
            ExperimentUtils.layout_title("R* \$\\Gamma_1\$ b=$(rs)pes"),
            all_plots[i]...;
            layout=layout, 
            size=(2000,1000)
        )

        if isnothing(name)
            println("Can't show multiple plots interactively.")
        else
            plt.savefig(allhist, "fig/$(name)_RG1_hist$(i).png")
        end
    end
end

function hist_RG1_fixgamma(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
)
    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval)

    gammas = [0,2,4,6,10]
    NBOUNDS = 6
    rbounds = LinRange(pes_bound, 1.5*pes_bound, NBOUNDS)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, 6)]
    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    maxval = 0
    all_plots = []

    for (i,gamma) in enumerate(gammas)
        inst1 = with_gamma1(inst, gamma)

        all_costs = []
        case_experiment(inst1, rbounds,
            with_rbound,
            LM.Model.create_R_model;
            optimize=RModel.optimizeRmodel!
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
            rstring,
            all_costs;
            mktitle=rb -> "Rbound = $(rb)pes",
            rescale=true
        )

        maxval = max(maxval, maximum(plt.ylims(plot)[end] for plot in plots))

        for (plot,rbound) in zip(plots, rbounds)
            plt.vline!(plot, [rbound], legend=false)
        end

        push!(all_plots, plots)
    end

    for (i,gamma) in enumerate(gammas)
        for plot in all_plots[i]
            plt.ylims!(plot, (0.0, maxval))
        end
        
        layout=@plt.layout [a{0.01h}; plt.grid(3,2)]

        allhist = plt.plot(
            ExperimentUtils.layout_title("R* \$\\Gamma_1=$(gamma)\$"),
            all_plots[i]...;
            layout=layout, 
            size=(2000,1000)
        )

        if isnothing(name)
            println("Can't show multiple plots interactively.")
        else
            plt.savefig(allhist, "fig/$(name)_RG1_hist_fixgamma$(i).png")
        end
    end
end

function budget_exceeding_RG1(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
)
    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval)

    gammas = 0:inst.T
    rbounds = LinRange(pes_bound, 1.5*pes_bound, 6)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, length(rbounds))]
    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    excplot = plt.Plot()
    for (i,rbound) in enumerate(rbounds)
        println("R* bound $(i)")
        inst1 = with_rbound(inst, rbound)
        exceeding = zeros(Int, length(gammas))

        case_experiment(inst1, gammas,
            with_gamma1,
            LM.Model.create_R_model;
            optimize=RModel.optimizeRmodel!,
        ) do j,_,xs,ys
            costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

            exceeding[j] = count(c -> c > rbound, costs)
        end

        probability = [c / N for c in exceeding]

        plt.plot!(excplot, gammas, probability;
            label="$(rstring[i])pes"
        )
    end

    exceeding = zeros(Int, length(gammas))
    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_pes_model
    ) do j,_,xs,ys
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

        exceeding[j] = count(c -> c > pes_bound, costs)
    end
    probability = [c / N for c in exceeding]

    plt.plot!(excplot, gammas, probability;
        label="minmax", linestyle=:dash)

    if isnothing(name)
        plt.gui(excplot)
    else
        plt.savefig(excplot, "fig/$(name)_RG1_exceeding.png")
    end
end #budget_exceeding_RG1

function solution_evolution(inst; name=nothing)
    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval)

    gammas = [0,2,4,6,8,10]
    NBOUNDS = 20
    rbounds = LinRange(pes_bound, 1.5*pes_bound, NBOUNDS)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, NBOUNDS)]

    allplots = []
    for (i,gamma) in enumerate(gammas)
        inst1 = with_gamma1(inst, gamma)
        results = zeros(Float64, NBOUNDS, inst.T)

        case_experiment(inst1, rbounds,
            with_rbound,
            RModel.create_R_model,
            optimize=RModel.optimizeRmodel!,
        ) do i,_,xs,_
            results[i,:] = collect(xs)
        end

        plot = plt.areaplot(rbounds, results,
            label=permutedims(["\$x_{$(i)}\$" for i in 1:inst.T]),
            palette=:Spectral_10,
            title="\$\\Gamma_1 = $(gamma)\$")

        push!(allplots, plot)
    end

    finalplot = plt.plot(allplots...,
        size=(2000,1000)
    )

    if isnothing(name)
        println("Can't show multiple plots interactively.")
    else
        plt.savefig(finalplot, "fig/$(name)_RG1_evolution.png")
    end
end
