include("ExperimentUtils.jl")
using .ExperimentUtils

import LotsizingModel as LM

import .LM.Hurwicz
import .LM.RModel

using .LM.Instance

import Plots as plt

using Accessors: @set,@reset
using Base.Threads
using JuMP
using Plots.PlotMeasures
using Printf
using ProgressMeter

function bicriterial(name, inst;
    alphas=LinRange(0,1,30),
    gammas=nothing,
)
    if isnothing(gammas)
        gammas = collect(0:inst.T)
    end

    inst_interval = with_interval(inst)

    plot = plt.Plot()
    plt_palette = plt.palette(:rainbow, length(gammas))

    @showprogress for gamma in gammas
        inst1 = with_gammaD(inst, gamma)

        mins = [0.0 for _ in alphas]
        maxs = [0.0 for _ in alphas]

        @threads for i in eachindex(alphas)
            alpha = alphas[i]
            model,x,y = Hurwicz.create_hurwicz_model(inst1, alpha)

            optimize!(model)

            xs = collect(JuMP.value.(x))
            ys = collect(JuMP.value.(y))

            cmin,cmax = get_cost_bounds(inst_interval, xs)
            mins[i] = cmin
            maxs[i] = cmax
        end

        plt.plot!(plot, mins, maxs,
            label="\$\\Gamma=$(gamma)\$",
            palette=plt_palette,
            markershape=:circle,
        )
    end

    plt.savefig(plot, "results/$(name)_hur_min-max.pdf")
    plt.savefig(plot, "results/$(name)_hur_min-max.png")
end


function pes_value(
    inst::LotSizingInstance;
    make_optimizer=LM.Utils.default_optimizer
)
    model, = LM.Model.create_minmax_model(inst;make_optimizer=make_optimizer)

    optimize!(model)

    return objective_value(model)
end

function bicriterialR(name, inst;
    alphas=LinRange(1.0,1.5,30),
    gammas=nothing,
)
    if isnothing(gammas)
        gammas = collect(0:inst.T)
    end

    inst_interval = with_interval(inst)
    pes_bound = pes_value(inst_interval)

    plot = plt.Plot()
    plt_palette = plt.palette(:rainbow, length(gammas))

    @showprogress for gamma in gammas
        inst1 = with_gammaD(inst, gamma)

        mins = [0.0 for _ in alphas]
        maxs = [0.0 for _ in alphas]

        @threads for i in eachindex(alphas)
            alpha = alphas[i]
            inst2 = with_rbound(inst1, alpha*pes_bound)
            model,x,y = RModel.create_R_model(inst2)

            optimize!(model)

            xs = collect(JuMP.value.(x))
            ys = collect(JuMP.value.(y))

            cmin,cmax = get_cost_bounds(inst_interval, xs)
            mins[i] = cmin
            maxs[i] = cmax
        end

        plt.plot!(plot, mins, maxs,
            label="\$\\Gamma=$(gamma)\$",
            palette=plt_palette,
            markershape=:circle,
        )
    end

    plt.savefig(plot, "results/$(name)_hurR_min-max.pdf")
    plt.savefig(plot, "results/$(name)_hurR_min-max.png")
end

function combined_hist_hurwicz(
    inst::LotSizingInstance,
    samples,
    ;
    name=nothing,
    gammas=nothing,
    alphas=LinRange(0,1,6),
    scale=nothing,
    scalefmt=nothing
)
    if isnothing(scale) != isnothing(scalefmt)
        throw("Both scale and scalefmt need to be provided")
    end

    if isnothing(gammas)
        gammas = 1:inst.T
    end

    N = length(samples)

    hcosts = []
    for alpha in alphas
        row_costs = []
        case_experiment(inst, gammas,
            with_gammaD,
            (i;kw...) -> Hurwicz.create_hurwicz_model(i, alpha),
        ) do _,_,xs,ys
            costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

            push!(row_costs, costs)
        end

        push!(hcosts, row_costs)
    end

    xmin = minimum(
        minimum(costs) for row in hcosts for costs in row
    )

    xmax = maximum(
        maximum(costs) for row in hcosts for costs in row
    )

    ## Plots
    all_plots = []
    ymax = 0

    # Sturges rule
    k = 1 + 2*ceil(Int64, log2(N))

    # Hurwicz plots
    for (i,alpha) in enumerate(alphas)
        plots = plot_histograms(
            xmin, xmax, k, N,
            gammas,
            hcosts[i];
            # only add gamma titles in first row (min-max)
            mktitle=i>1 ? nothing : gamma -> "\$\\Gamma_1 = $(gamma)\$",
            rescale=true,
            perc90=true,
            hideyaxis=true,
        )

        plt.ylabel!(plots[begin], "\\alpha=$(@sprintf "%.2f" alpha)")
        plt.plot!(plots[begin], left_margin = 25mm)
        ymax = max(ymax, maximum(plt.ylims(plot)[end] for plot in plots))

        append!(all_plots, plots)
    end

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
        plt.grid(length(alphas), length(gammas))
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
        plt.savefig(allhist, "results/$(name)_hur_combined_hist.pdf")
        plt.savefig(allhist, "results/$(name)_hur_combined_hist.png")
    end
end #combined_hist_hurwicz

function experiment(name, path)
    println("# Experiment \"$(name)\"")
    inst = LM.load_instance(path)

    N = 10000
    println("### Preparing sample lead times")
    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    println("### Preparing Hurwicz histograms")
        combined_hist_hurwicz(inst, samples;
            name=name,
            gammas=[2,4,6,10],
            scale=1e5,
            scalefmt="\$\\cdot 10^5\$",
        )

    println("### Preparing bicriterial Hurwicz plots")
        bicriterial(name, inst)
    println("### Preparing bicriterial R* plots")
        bicriterialR(name, inst)
end

function run(;make_optimizer=create_gurobi)
    if !isnothing(make_optimizer)
        LM.Utils.set_default_optimizer!(make_optimizer)
    end

    mkpath("results")

    println("Results will be placed in the directory `results`")

    experiment("10_21_1", "data/gen/10_21_1.dat")
    # experiment("15_31_1", "data/gen/15_31_1.dat")
end