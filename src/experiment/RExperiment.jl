import LotsizingModel as LM

import .LM.RModel
import .LM.MinMaxModel
import .LM.MinMinModel
import .LM.Deterministic
using .LM.Instance

include("ExperimentUtils.jl")
using .ExperimentUtils

import Plots as plt
using Plots.PlotMeasures
using Accessors: @set,@reset
using DelimitedFiles
using JuMP
using Printf
using ProgressMeter
using StatsBase

function opt_value(
    inst::LotSizingInstance;
    make_optimizer=LM.Utils.default_optimizer
)
    model, = LM.Model.create_minmin_model(inst; make_optimizer=make_optimizer)

    optimize!(model)

    return objective_value(model)
end

function pes_value(
    inst::LotSizingInstance;
    make_optimizer=LM.Utils.default_optimizer
)
    model, = LM.Model.create_minmax_model(inst;make_optimizer=make_optimizer)

    optimize!(model)

    return objective_value(model)
end


function combined_hist_RG1(
    inst::LotSizingInstance,
    samples,
    ;
    name=nothing,
    gammas=nothing,
    scale=nothing,
    scalefmt=nothing,
    make_optimizer=LM.Utils.default_optimizer
)
    if isnothing(scale) != isnothing(scalefmt)
        throw("Both scale and scalefmt need to be provided")
    end

    if isnothing(gammas)
        gammas = 1:inst.T
    end

    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval; make_optimizer)

    NBOUNDS = 5
    rbounds = LinRange(pes_bound, 1.5*pes_bound, NBOUNDS)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, NBOUNDS)]

    N = length(samples)

    all_plots = []
    ymax = 0

    pes_costs = []
    pes_stats = []
    case_experiment(inst, gammas,
        with_gammaD,
        LM.Model.create_minmax_model,
        make_optimizer=create_gurobi,
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
            with_gammaD,
            LM.Model.create_R_model;
            optimize=RModel.optimizeRmodel!,
            make_optimizer=create_gurobi,
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
        with_gammaD,
        LM.Model.create_minmin_model,
        make_optimizer=create_gurobi,
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
    det_model,x,y = Deterministic.create_deterministic(inst; make_optimizer=make_optimizer)
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

function pareto_plot(
    plot, points,
    ;
    label,
    marker_z,
    markershape,
)
    xs,ys = collect.(zip(points...))
    plt.plot!(plot, xs, ys, seriestype=:scatter,
        marker_z=marker_z,
        markershape=markershape,
        markersize=8,
        markerstrokewidth=1,
        label=label,
        color=:Spectral,
    )
end

function plot_front(
    plot, all_points;
)
    front = ExperimentUtils.pareto_front(all_points)
    xs,ys = collect.(zip(front...))
    plt.plot!(plot, xs, ys,
        linestyle=:dash,
        label="Pareto front",
        color=:black,
        z_order=:back,
    )
end

function pareto_front(
    inst::LotSizingInstance,
    samples,
    ;
    name=nothing,
    scale=nothing,
    scalefmt=nothing,
    make_optimizer=LM.Utils.default_optimizer,
)
    if isnothing(scale) != isnothing(scalefmt)
        throw("Both scale and scalefmt need to be provided")
    end
    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    pes_bound = pes_value(inst_interval; make_optimizer=make_optimizer)

    gammas = 0:inst.T
    NBOUNDS = 3
    rbounds = LinRange(pes_bound, 1.5*pes_bound, NBOUNDS)
    rstring = [@sprintf "%.2f" rb for rb in LinRange(1.0, 1.5, NBOUNDS)]
  
    all_min_max = []
    all_med_quant = []
    all_mean_stdev = []

    min_max_points = []
    med_quant_points = []
    mean_stddev_points = []
    case_experiment(inst, gammas,
        with_gammaD,
        MinMaxModel.create_minmax_model,
        make_optimizer=make_optimizer,
    ) do _,_,xs,ys
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]
        stats = ExperimentUtils.experiment_stats(costs)

        min_max = (stats.min, stats.max)
        med_quant = (stats.median, stats.perc90)
        mean_stddev = (stats.mean, stats.stddev)

        if !isnothing(scale)
            min_max = min_max ./ scale
            med_quant = med_quant ./ scale
            mean_stddev = mean_stddev ./ scale
        end

        push!(min_max_points, min_max)
        push!(med_quant_points, med_quant)
        push!(mean_stddev_points, mean_stddev)
    end
    append!(all_min_max, min_max_points)
    append!(all_med_quant, med_quant_points)
    append!(all_mean_stdev, mean_stddev_points)
    
    minmaxplot = plt.Plot()
    medquantplot = plt.Plot()
    meanstddevplot = plt.Plot()

    batch = [
        (minmaxplot, min_max_points),
        (medquantplot, med_quant_points),
        (meanstddevplot, mean_stddev_points),
    ]
    for (plot, points) in batch
        pareto_plot(plot, points;
            label="Min-Max",
            markershape=:diamond,
            marker_z=gammas,
        )
    end

    min_max_points = []
    med_quant_points = []
    mean_stddev_points = []
    case_experiment(inst, gammas,
        with_gammaD,
        MinMinModel.create_minmin_model,
        make_optimizer=make_optimizer,
    ) do _,_,xs,ys
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]
        stats = ExperimentUtils.experiment_stats(costs)

        min_max = (stats.min, stats.max)
        med_quant = (stats.median, stats.perc90)
        mean_stddev = (stats.mean, stats.stddev)

        if !isnothing(scale)
            min_max = min_max ./ scale
            med_quant = med_quant ./ scale
            mean_stddev = mean_stddev ./ scale
        end

        push!(min_max_points, min_max)
        push!(med_quant_points, med_quant)
        push!(mean_stddev_points, mean_stddev)
    end
    append!(all_min_max, min_max_points)
    append!(all_med_quant, med_quant_points)
    append!(all_mean_stdev, mean_stddev_points)

    batch = [
        (minmaxplot, min_max_points),
        (medquantplot, med_quant_points),
        (meanstddevplot, mean_stddev_points),
    ]
    for (plot, points) in batch
        pareto_plot(plot, points;
            label="Min-Min",
            markershape=:square,
            marker_z=gammas,
        )
    end

    RMARKERS = [:circle, :hexagon, :star4, :utriangle]

    for (i,rbound) in enumerate(rbounds)
        inst1 = with_rbound(inst, rbound)

        min_max_points = []
        med_quant_points = []
        mean_stddev_points = []
        case_experiment(inst1, gammas,
            with_gammaD,
            RModel.create_R_model,
            optimize=RModel.optimizeRmodel!,
            make_optimizer=make_optimizer,
        ) do _,_,xs,ys
            costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]
            stats = ExperimentUtils.experiment_stats(costs)

            min_max = (stats.min, stats.max)
            med_quant = (stats.median, stats.perc90)
            mean_stddev = (stats.mean, stats.stddev)

            if !isnothing(scale)
                min_max = min_max ./ scale
                med_quant = med_quant ./ scale
                mean_stddev = mean_stddev ./ scale
            end

            push!(min_max_points, min_max)
            push!(med_quant_points, med_quant)
            push!(mean_stddev_points, mean_stddev)
        end
        append!(all_min_max, min_max_points)
        append!(all_med_quant, med_quant_points)
        append!(all_mean_stdev, mean_stddev_points)


        batch = [
            (minmaxplot, min_max_points),
            (medquantplot, med_quant_points),
            (meanstddevplot, mean_stddev_points),
        ]
        for (plot, points) in batch
            pareto_plot(plot, points;
                label="R*\$($(rstring[i]))\$",
                markershape=RMARKERS[i],
                marker_z=gammas,
            )
        end
    end

    batch = [
        (minmaxplot, all_min_max),
        (medquantplot, all_med_quant),
        (meanstddevplot, all_mean_stdev),
    ]
    for (plot, all_points) in batch
        plot_front(plot, all_points)
    end

    batch = [
        (minmaxplot, "Minimal cost", "Maximal cost"),
        (medquantplot, "Median cost", "0.9-quantile"),
        (meanstddevplot, "Mean cost", "Standard deviation"),
    ]

    for (plot, xname, yname) in batch
        xlabel = xname * (isnothing(scale) ? "" : " $(scalefmt)")
        plt.xlabel!(plot, xlabel)
        ylabel = yname * (isnothing(scale) ? "" : " $(scalefmt)") 
        plt.ylabel!(plot, ylabel)
        plt.plot!(plot,
            size=(1000,750),
            left_margin=5mm,
            legend_font_pointsize=12,
            tickfontsize=12,
            guidefontsize=14,
            colorbar_title="\$\\Gamma^d\$",
            colorbar_titlefontsize=14,
        )
    end

    if isnothing(name)
        println("Can't show multiple plots interactively.")
    else
        plt.savefig(minmaxplot, "results/$(name)_min-max_pareto.pdf")
        plt.savefig(medquantplot, "results/$(name)_med-quant_pareto.pdf")
        plt.savefig(meanstddevplot, "results/$(name)_mean-stddev_pareto.pdf")
    end
end

function experiments(name,path;
    histgammas=nothing,
    make_optimizer=LM.Utils.default_optimizer,
)
    println("# Experiment \"$(name)\"")
    inst = LM.load_instance(path)

    N = 10000
    println("### Preparing sample arrivals")
    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    println("### Combined hist G1")
        combined_hist_RG1(inst, samples,
            name=name,
            gammas=histgammas,
            scale=1e5,
            scalefmt="\$\\cdot 10^5\$",
            make_optimizer=make_optimizer,
        )

    println("### Pareto front")
        pareto_front(inst, samples,
            name=name,
            scale=1e5,
            scalefmt="\$\\cdot 10^5\$",
            make_optimizer=make_optimizer,
        )
    
    println("Done.")
end

function run(;make_optimizer=LM.Utils.default_optimizer)
    mkpath("results")

    experiments("10_21_1", "data/gen/10_21_1.dat",
        make_optimizer=make_optimizer,
        histgammas=[2,4,6,10],
    )
    experiments("15_31_1", "data/gen/15_31_1.dat",
        make_optimizer=make_optimizer,
        histgammas=[2,5,8,10,13],
    )
end #run
