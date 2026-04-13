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

    minbound = zeros(length(gammas))
    maxbound = zeros(length(gammas))

    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_pes_model
    ) do i,_,xs,ys
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

        pes_min[i] = minimum(costs)
        pes_max[i] = maximum(costs)

        inst1 = with_interval(inst)
        minbound[i],maxbound[i] = get_cost_bounds(inst1, xs)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, gammas, minbound;
        linestyle=:dot,label="max bound")
    plt.plot!(cost_plot, gammas, minbound;
        linestyle=:dot,label="max bound")
    plt.plot!(cost_plot, gammas, pes_min;
        markershape=:circle, label="min sample")
    plt.plot!(cost_plot, gammas, pes_max;
        markershape=:circle,label="max sample")
    
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

    minbound = zeros(length(gammas))
    maxbound = zeros(length(gammas))

    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    case_experiment(inst, gammas,
        with_gamma2,
        LM.Model.create_pes_model,
    ) do i,_,xs,ys
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

        pes_min[i] = minimum(costs)
        pes_max[i] = maximum(costs)

        inst1 = with_interval(inst)
        minbound[i],maxbound[i] = get_cost_bounds(inst1, xs)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, gammas, minbound;
        linestyle=:dot,label="min bound")
    plt.plot!(cost_plot, gammas, maxbound;
        linestyle=:dot,label="max bound")
    plt.plot!(cost_plot, gammas, pes_min;
        markershape=:circle,label="min sample")
    plt.plot!(cost_plot, gammas, pes_max;
        markershape=:circle,label="max sample")
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

    minbound = zeros(length(gammas))
    maxbound = zeros(length(gammas))

    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    case_experiment(inst, gammas,
        with_gamma1,
        LM.Model.create_opt_model
    ) do i,_,xs,ys
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

        pes_min[i] = minimum(costs)
        pes_max[i] = maximum(costs)

        inst1 = with_interval(inst)
        minbound[i],maxbound[i] = get_cost_bounds(inst1, xs)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, gammas, minbound;
        linestyle=:dot,label="min bound")
    plt.plot!(cost_plot, gammas, maxbound;
        linestyle=:dot,label="max bound")
    plt.plot!(cost_plot, gammas, pes_min;
        markershape=:circle, label="min sample")
    plt.plot!(cost_plot, gammas, pes_max;
        markershape=:circle,label="max sample")
    
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

    minbound = zeros(length(gammas))
    maxbound = zeros(length(gammas))

    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

    case_experiment(inst, gammas,
        with_gamma2,
        LM.Model.create_opt_model
    ) do i,_,xs,ys
        costs = [prod_lead_cost(inst,xs,ys,sample) for sample in samples]

        pes_min[i] = minimum(costs)
        pes_max[i] = maximum(costs)

        inst1 = with_interval(inst)
        minbound[i],maxbound[i] = get_cost_bounds(inst1, xs)
    end

    cost_plot = plt.Plot()
    plt.plot!(cost_plot, gammas, minbound;
        linestyle=:dot,label="min bound")
    plt.plot!(cost_plot, gammas, maxbound;
        linestyle=:dot,label="max bound")
    plt.plot!(cost_plot, gammas, pes_min;
        markershape=:circle, label="min sample")
    plt.plot!(cost_plot, gammas, pes_max;
        markershape=:circle,label="max sample")
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
    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

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
        rescale=true,
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
    samples = ExperimentUtils.rejection_sample_arrivals(inst, N)

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
        rescale=true,
    )

    allhist = plt.plot(plots...; layout=length(gammas), size=(2000,1000))

    if isnothing(name)
        plt.gui(allhist)
    else
        plt.savefig("fig/$(name)_optG1_hist.png")
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
    experiments("10_41_1", "data/gen/10_41_1.dat")
end #run
