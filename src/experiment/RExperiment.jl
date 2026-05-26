import LotsizingModel as LM

import .LM.StatUtils
import .LM.RModel
import .LM.PesModel
import .LM.OptModel
import .LM.Deterministic
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


function combined_hist_RG1(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
    gammas=nothing,
    scale=nothing,
    scalefmt=nothing,
)
    if isnothing(scale) != isnothing(scalefmt)
        throw("Both scale and scalefmt need to be provided")
    end

    if isnothing(gammas)
        gammas = [2,4,6,10]
    end

    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval)

    NBOUNDS = 5
    rbounds = LinRange(pes_bound, 1.5*pes_bound, NBOUNDS)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, NBOUNDS)]
    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    all_plots = []
    ymax = 0

    pes_costs = []
    pes_stats = []
    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_pes_model
    ) do _,_,xs,ys
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]
        stats = ExperimentUtils.experiment_stats(costs)
        push!(pes_stats, stats)

        push!(pes_costs, costs)
    end

    r_costs = []
    r_stats = []
    for rbound in rbounds
        inst1 = with_rbound(inst, rbound)

        row_costs = []
        row_stats = []
        case_experiment(inst1, gammas,
            with_gamma1,
            LM.Model.create_R_model;
            optimize=RModel.optimizeRmodel!
        ) do _,_,xs,ys
            costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]
            stats = ExperimentUtils.experiment_stats(costs)
            push!(row_stats, stats)

            push!(row_costs, costs)
        end

        push!(r_stats, row_stats)

        push!(r_costs, row_costs)
    end

    opt_costs = []
    opt_stats = []
    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_opt_model
    ) do _,_,xs,ys
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]
        stats = ExperimentUtils.experiment_stats(costs)
        push!(opt_stats, stats)

        push!(opt_costs, costs)
    end

    xmin = min(
        minimum(minimum(costs) for costs in pes_costs),
        minimum(minimum(costs) for costs in opt_costs),
        minimum(minimum(costs) for row in r_costs for costs in row),
    )

    xmax = max(
        maximum(maximum(costs) for costs in pes_costs),
        maximum(maximum(costs) for costs in opt_costs),
        maximum(maximum(costs) for row in r_costs for costs in row),
    )

    ## Stats table

    table_rows = []

    # Det problem for stats
    det_model,x,y = Deterministic.create_deterministic(inst)
    JuMP.optimize!(det_model)
    xs = collect(JuMP.value.(x))
    ys = collect(JuMP.value.(y))
    det_costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]
    det_stats = ExperimentUtils.experiment_stats(det_costs)

    line = ExperimentUtils.stats_table_row(det_stats, "-", model="\$\\textsc{Det}\$\n")
    push!(table_rows, line)
    push!(table_rows, "\\hline")
    push!(table_rows, "\\hline")

    for (i,row) in enumerate(pes_stats)
        line = ExperimentUtils.stats_table_row(row, gammas[i],
            model=i>1 ? nothing : "\$\\textsc{Min-Max}\$\n")

        push!(table_rows, line)
    end
    push!(table_rows, "\\hline")
    push!(table_rows, "\\hline")

    for (j, row_stats) in enumerate(r_stats)
        for (i,row) in enumerate(row_stats)
            line = ExperimentUtils.stats_table_row(row, gammas[i],
                model=i>1 ? nothing : "\$R^*($(rstring[j]))\$\n"
            )

            push!(table_rows, line)
        end

        push!(table_rows, "\\hline")
        push!(table_rows, "\\hline")
    end

    for (i,row) in enumerate(opt_stats)
        line = ExperimentUtils.stats_table_row(row, gammas[i],
            model=(i>1 ? nothing : "\$\\textsc{Min-Min}\$\n"),
            skipnl=(i==length(opt_stats)),
        )

        push!(table_rows, line)
    end

    table_path = "results/$(name)_combined_stats.txt"
    table_content = join(table_rows, "\n")
    open(table_path, "w") do file
        write(file, table_content)
    end

    ## Plots

    # Sturges rule
    k = 1 + 2*ceil(Int64, log2(N))

    # Min-max plots
    pes_plots = plot_histograms(
        xmin, xmax, k, N, 
        gammas,
        pes_costs;
        mktitle=gamma -> "\$\\Gamma^d = $(gamma)\$",
        rescale=true,
        perc90=true,
        hideyaxis=true,
    )

    plt.ylabel!(pes_plots[begin], "Min-Max")
    plt.plot!(pes_plots[begin], left_margin = 25mm)
    ymax = max(ymax, maximum(plt.ylims(plot)[end] for plot in pes_plots))

    append!(all_plots, pes_plots)

    # R* plots
    R_plots = []
    for (i,rbound) in enumerate(rbounds)
        plots = plot_histograms(
            xmin, xmax, k, N,
            gammas,
            r_costs[i];
            # only add gamma titles in first row (min-max)
            # mktitle=gamma -> "\$\\Gamma_1 = $(gamma)\$",
            rescale=true,
            perc90=true,
            hideyaxis=true,
        )

        for plot in plots
            plt.vline!(plot, [rbound],
                color=:red,
                linewidth=3,
                legend=false,
            )
        end

        plt.ylabel!(plots[begin], "R* ($(rstring[i]))")
        plt.plot!(plots[begin], left_margin = 25mm)
        ymax = max(ymax, maximum(plt.ylims(plot)[end] for plot in plots))

        append!(all_plots, plots)
        push!(R_plots, plots)
    end

    # Min-min plots
    opt_plots = plot_histograms(
        xmin, xmax, k, N,
        gammas,
        opt_costs;
        # only add gamma titles in first row (min-max)
        # mktitle=gamma -> "\$\\Gamma_1 = $(gamma)\$",
        rescale=true,
        perc90=true,
        hideyaxis=true,
    )

    plt.ylabel!(opt_plots[begin], "Min-Min")
    plt.plot!(opt_plots[begin], left_margin = 25mm)
    if !isnothing(scale)
        for plot in opt_plots
            plt.xlabel!(plot, "Cost $(scalefmt)")
            plt.plot!(plot, bottom_margin = 20mm)
        end
    end
    ymax = max(ymax, maximum(plt.ylims(plot)[end] for plot in opt_plots))

    append!(all_plots, opt_plots)

    # Final processing
    for plot in all_plots
        plt.ylims!(plot, (0.0, ymax))
        plt.xlims!(plot, (xmin, xmax))
        plt.plot!(plot,
            titlefontsize=30,
            yguidefontsize=30,
            xguidefontsize=30,
            tickfontsize=26,
            ytick_direction=:in,
            xformatter=(isnothing(scale) ? nothing : 
                x -> x / scale)
        )
    end

    layout= @plt.layout [
        plt.grid(2+length(rbounds), length(gammas))
    ]

    allhist = plt.plot(
        all_plots...,
        ;
        layout=layout, 
        size=(3000,3000),
        plot_titlefontsize=24,
    )

    if isnothing(name)
        println("Can't show multiple plots interactively.")
    else
        plt.savefig(allhist, "results/$(name)_RG1_combined_hist.pdf")
    end
end #combined_hist_RG1

function pareto_front(
    inst::LotSizingInstance,
    ;
    name=nothing,
    scale=nothing,
    scalefmt=nothing,
)
    if isnothing(scale) != isnothing(scalefmt)
        throw("Both scale and scalefmt need to be provided")
    end
    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval)

    gammas = 0:inst.T
    NBOUNDS = 3
    rbounds = LinRange(pes_bound, 1.5*pes_bound, NBOUNDS)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, NBOUNDS)]

    plot = plt.Plot()
    all_points = []

    points = []
    case_experiment(inst, gammas,
        with_gamma1,
        PesModel.create_pes_model
    ) do _,_,xs,_
        res = ExperimentUtils.get_cost_bounds(inst, collect(xs))

        if !isnothing(scale)
            res = res ./ scale
        end

        push!(points, res)
    end
    append!(all_points, points)

    xs,ys = collect.(zip(points...))
    plt.plot!(plot, xs, ys, seriestype=:scatter,
        marker_z=gammas,
        markershape=:diamond,
        markersize=8,
        markerstrokewidth=1,
        label="Min-Max",
        color=:Spectral,
    )

    points = []
    case_experiment(inst, gammas,
        with_gamma1,
        OptModel.create_opt_model
    ) do _,_,xs,_
        res = ExperimentUtils.get_cost_bounds(inst, collect(xs))

        if !isnothing(scale)
            res = res ./ scale
        end

        push!(points, res)
    end
    append!(all_points, points)

    xs,ys = collect.(zip(points...))
    plt.plot!(plot, xs, ys, seriestype=:scatter,
        marker_z=gammas,
        markershape=:square,
        markersize=8,
        markerstrokewidth=1,
        label="Min-Min",
        color=:Spectral,
    )

    RMARKERS = [:circle, :hexagon, :star4, :utriangle]

    for (i,rbound) in enumerate(rbounds)
        inst1 = with_rbound(inst, rbound)

        points = []
        case_experiment(inst1, gammas,
            with_gamma1,
            RModel.create_R_model,
            optimize=RModel.optimizeRmodel!,
        ) do _,_,xs,_
            res = ExperimentUtils.get_cost_bounds(inst, collect(xs))

            if !isnothing(scale)
                res = res ./ scale
            end

            push!(points, res)
        end
        append!(all_points, points)

        xs,ys = collect.(zip(points...))
        plt.plot!(plot, xs, ys, seriestype=:scatter,
            label="R*\$($(rstring[i]))\$",
            marker_z=gammas,
            markershape=RMARKERS[i],
            markersize=8,
            markerstrokewidth=1,
            color=:Spectral,
        )
    end

    front = ExperimentUtils.pareto_front(all_points)
    xs,ys = collect.(zip(front...))
    plt.plot!(plot, xs, ys,
        linestyle=:dash,
        label="Pareto front",
        color=:black,
        z_order=:back,
    )

    ylabel = "Maximal cost" * (isnothing(scale) ? "" : " $(scalefmt)") 
    plt.ylabel!(plot, ylabel)
    xlabel = "Minimal cost" * (isnothing(scale) ? "" : " $(scalefmt)")
    plt.xlabel!(plot, xlabel)
    plt.plot!(plot,
        size=(1000,750),
        left_margin=5mm,
        legend_font_pointsize=12,
        tickfontsize=12,
        guidefontsize=14,
        colorbar_title="\$\\Gamma^d\$",
        colorbar_titlefontsize=14,
    )

    if isnothing(name)
        println("Can't show multiple plots interactively.")
    else
        plt.savefig(plot, "results/$(name)_RG1_pareto.pdf")
    end
end

function experiments(name,path; skip_long=false, histgammas=nothing)
    println("# Experiment \"$(name)\"")
    inst = LM.load_instance(path)

    N = 10000

    if !skip_long
    println("### Combined hist G1")
        combined_hist_RG1(inst, name=name, N=N,
            gammas=histgammas,
            scale=1e5,
            scalefmt="\$\\cdot 10^5\$")
    end

    # println("### Pareto front")
    #     pareto_front(inst, name=name,
    #         scale=1e5,
    #         scalefmt="\$\\cdot 10^5\$")
    
    println("Done.")
end

function run()
    mkpath("results")

    experiments("10_21_1", "data/gen/10_21_1.dat")
    experiments("15_31_1", "data/gen/15_31_1.dat",
        histgammas=[2,5,8,10,13],
    )
end #run
