import LotsizingModel as LM

import .LM.StatUtils
using .LM.Instance
using .LM.ExperimentUtils

import Plots as plt
using Accessors: @set,@reset
using ProgressMeter
using JuMP
using Statistics

function cost_pesG1(inst::LotSizingInstance;name=nothing)
    gammas = 0:inst.T
    results = zeros(length(gammas))

    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_pes_model
    ) do i,model,_,_
        results[i] = objective_value(model)
    end

    cost_plot = make_cost_plot(gammas, results;
        title="Min-max cost \$\\Gamma_1\$",
        ylabel="cost",
        xlabel="\$\\Gamma\$",
    )

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig("fig/$(name)_pesG1_cost.png")
    end
end

function cost_pesG2(inst::LotSizingInstance; name=nothing)
    gammas = 0:max_gamma2_budget(inst)
    results = zeros(length(gammas))

    case_experiment(inst, gammas,
        with_gamma2,
        LM.Model.create_pes_model
    ) do i,model,_,_
        results[i] = objective_value(model)
    end

    cost_plot = make_cost_plot(gammas, results;
        title="Min-max cost \$\\Gamma_2\$",
        ylabel="cost",
        xlabel="\$\\Gamma\$",
    )

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig("fig/$(name)_pesG2_cost.png")
    end
end

function cost_optG1(inst::LotSizingInstance;name=nothing)
    gammas = 0:inst.T
    results = zeros(length(gammas))

    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_opt_model
    ) do i,model,_,_
        results[i] = objective_value(model)
    end

    cost_plot = make_cost_plot(gammas, results;
        title="Min-min cost \$\\Gamma_1\$",
        ylabel="cost",
        xlabel="\$\\Gamma\$",
    )

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig("fig/$(name)_optG1_cost.png")
    end
end

function cost_optG2(inst::LotSizingInstance; name=nothing)
    gammas = 0:max_gamma2_budget(inst)
    results = zeros(length(gammas))

    case_experiment(inst, gammas,
        with_gamma2,
        LM.Model.create_opt_model
    ) do i,model,_,_
        results[i] = objective_value(model)
    end

    cost_plot = make_cost_plot(gammas, results;
        title="Min-min cost \$\\Gamma_2\$",
        ylabel="cost",
        xlabel="\$\\Gamma\$",
    )

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig("fig/$(name)_optG2_cost.png")
    end
end

function sample_pesG1(inst::LotSizingInstance; N=1000, name=nothing)
    gammas = 0:inst.T
    pes_min = zeros(length(gammas))
    pes_max = zeros(length(gammas))

    samples = sample_arrivals(inst, N)

    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_pes_model
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
    plt.title!(cost_plot, "Min-max sampled cost \$\\Gamma_1\$")
    plt.ylabel!(cost_plot, "cost")
    plt.xlabel!(cost_plot, "\$\\Gamma\$")

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig(cost_plot,"fig/$(name)_pesG1_sample.png")
    end
end

function sample_pesG2(inst::LotSizingInstance;N=1000,name=nothing)
    gammas = 0:max_gamma2_budget(inst)
    pes_min = zeros(length(gammas))
    pes_max = zeros(length(gammas))

    samples = sample_arrivals(inst, N)

    case_experiment(inst, gammas,
        with_gamma2,
        LM.Model.create_pes_model
    ) do i,_,xs,ys
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

        pes_min[i] = minimum(costs)
        pes_max[i] = maximum(costs)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, gammas, pes_min;
        markershape=:circle,label="min")
    plt.plot!(cost_plot, gammas, pes_max;
        markershape=:circle,label="max")
    plt.title!(cost_plot, "Min-max sampled cost \$\\Gamma_2\$")

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig("fig/$(name)_pesG2_sample.png")
    end
end

function sample_optG1(inst::LotSizingInstance; N=1000, name=nothing)
    gammas = 0:inst.T
    pes_min = zeros(length(gammas))
    pes_max = zeros(length(gammas))

    samples = sample_arrivals(inst, N)

    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_opt_model
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
    plt.title!(cost_plot, "Min-min sampled cost \$\\Gamma_1\$")
    plt.ylabel!(cost_plot, "cost")
    plt.xlabel!(cost_plot, "\$\\Gamma\$")

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig(cost_plot,"fig/$(name)_optG1_sample.png")
    end
end

function sample_optG2(inst::LotSizingInstance; N=1000, name=nothing)
    gammas = 0:max_gamma2_budget(inst)
    pes_min = zeros(length(gammas))
    pes_max = zeros(length(gammas))

    samples = sample_arrivals(inst, N)

    case_experiment(inst, gammas,
        with_gamma2,
        LM.Model.create_opt_model
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
    plt.title!(cost_plot, "Min-min sampled cost \$\\Gamma_2\$")
    plt.ylabel!(cost_plot, "cost")
    plt.xlabel!(cost_plot, "\$\\Gamma\$")

    if isnothing(name)
        plt.gui(cost_plot)
    else
        plt.savefig(cost_plot,"fig/$(name)_optG2_sample.png")
    end
end

function hist_pesG1(
    inst::LotSizingInstance
    ;
    N=100,
    name=nothing,
)
    gammas = 0:inst.T
    samples = sample_arrivals(inst, N)

    all_costs = []

    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_pes_model
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

    allhist = plt.plot(plots...; layout=length(gammas), size=(2000,1000))

    if isnothing(name)
        plt.gui(allhist)
    else
        plt.savefig("fig/$(name)_pesG1_hist.png")
    end
end

function hist_optG1(
    inst::LotSizingInstance
    ;
    N=100,
    name=nothing,
)
    gammas = 0:inst.T
    samples = sample_arrivals(inst, N)

    all_costs = []

    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_opt_model
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

    allhist = plt.plot(plots...; layout=length(gammas), size=(2000,1000))

    if isnothing(name)
        plt.gui(allhist)
    else
        plt.savefig("fig/$(name)_optG1_hist.png")
    end
end

function heatmapG1(
    inst::LotSizingInstance
    ;
    N=100,
    name=nothing,
    force=false,
)
    gammas = 0:inst.T

    mkpath("output")
    tmpname = isnothing(name) ? "TMP" : name
    SAVEFILE="output/$(tmpname)_heatmapG1_$(N).csv"
    all_costs = StatUtils.with_file_cache(SAVEFILE; force=force) do 
        samples = sample_arrivals(inst, N)
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

    cont = plt.contourf(cutoffs[begin:end-1], gammas, hist,
        levels=30, 
        color=plt.cgrad(:turbo, scale=:exp), lw=0)
    plt.title!(cont, "Simulated cost (minmax model)")
    plt.xlabel!(cont, "cost")
    plt.ylabel!(cont, "\$\\Gamma_1\$")

    if isnothing(name)
        plt.gui(cont)
    else
        plt.savefig("fig/$(name)_pesG1_histcont.png")
    end
end

function experiments(name, path)
    println("# Experiment \"$(name)\"")
    inst = LM.load_instance(path)

    N = 10000

    # MinMax
    println("## Min-max experiments")
    println("### Cost Gamma1")
        cost_pesG1(inst;name=name)
    println("### Cost Gamma2")
        cost_pesG2(inst;name=name)
    
    println("### Sample Gamma1")
        sample_pesG1(inst;name=name,N=N)
    println("### Sample Gamma2")
        sample_pesG2(inst;name=name,N=N)

    println("### Histogram Gamma1")
        hist_pesG1(inst;name=name,N=N)
    
    println("## Min-min experiments")
    println("### Cost Gamma1")
        cost_optG1(inst;name=name)
    println("### Cost Gamma2")
        cost_optG2(inst;name=name)

    println("### Sample Gamma1")
        sample_optG1(inst;name=name,N=N)
    println("### Sample Gamma2")
        sample_optG2(inst;name=name,N=N)


    println("### Histogram Gamma1")
        hist_optG1(inst;name=name,N=N)

end #experiments

function run()
    mkpath("fig")
    experiments("10_21_1", "data/gen/10_21_1.dat")
    # experiments("10_41_1", "data/gen/10_41_1.dat")
end #run
