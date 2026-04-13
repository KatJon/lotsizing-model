__precompile__(true)
module ExperimentUtils

import ..StatUtils
import ..OptModel
import ..PesModel
using ..Instance

import Plots as plt
import JuMP

using Plots.PlotMeasures
using Accessors: @set,@reset
using ProgressMeter: @showprogress
using Random

export with_interval,
    with_gamma1, with_gamma2,
    with_rbound
export case_experiment
export plot_histograms
export make_cost_plot
export get_cost_bounds

"Create new instance from `inst` with unlimited budget"
function with_interval(inst::LotSizingInstance)
    return @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
end

"Create new instance from `inst` with Gamma1 budget set to `gamma`"
function with_gamma1(inst::LotSizingInstance, gamma)
    budget = Uncertainty.Model(Uncertainty.Gamma1, gamma)
    
    return @set inst.uncertainty=budget
end

"Create new instance from `inst` with Gamma2 budget set to `gamma`"
function with_gamma2(inst::LotSizingInstance, gamma)
    budget = Uncertainty.Model(Uncertainty.Gamma2, gamma)
    
    return @set inst.uncertainty=budget
end

"Create new instance from `inst` with R* bound set to `rbound`"
function with_rbound(inst::LotSizingInstance, rbound)
    return @set inst.rbound=rbound
end

"""
Perform experiment by cases.

Arguments:
* save_results - function that takes index of case, model, values of x, and values of y
* inst - LotSizingInstance
* cases - vector of cases to consider
* update_inst - function to create instance with case_param (e.g. `with_gamma1`)
* create_model - function creating model, vector of x variables, and vector of y variables

Optional arguments:
* after_solve - function called after model was solved
* optimize - alternative implementation for optimize (may return a new model and x,y variables)
"""
function case_experiment(
    save_results,
    inst::LotSizingInstance,
    cases,
    update_inst,
    create_model,
    ;
    after_solve=nothing,
    optimize=nothing
)
    @showprogress for (i,case_param) in enumerate(cases)
        inst1 = update_inst(inst, case_param)
        model,x,y, = create_model(inst1)

        if !isnothing(optimize)
            model,x,y = optimize(inst1,model,x,y)
        else
            JuMP.optimize!(model)
        end
        if !JuMP.is_solved_and_feasible(model)
            error("Model ($(i),$(case_param)) cannot be solved, reason=$(JuMP.termination_status(model))")
            continue
        end

        xs = collect(JuMP.value.(x))
        ys = collect(JuMP.value.(y))

        if !isnothing(after_solve)
            after_solve(model,xs,ys)
        end

        save_results(i,model,xs,ys)
    end
end #case_experiment

"""
Plot histograms for gathered data.

Arguments:
* amin, amax - minimal and maximal values in `all_data`,
* nbuckets - number of buckets,
* nsamples - number of sample points,
* cases - vector of cases to plot,
* all_data - sampled data.

Optional arguments:
* mktitle - function to set title based on case parameter
* rescale - (default: false) rescale all plots to match the biggest bar

Returns: vector of plots
"""
function plot_histograms(
    amin, amax, nbuckets, nsamples,
    cases,
    all_data, 
    ; 
    mktitle=nothing,
    rescale=false,
)
    plots=[]

    maxbin = 0.0

    for (param, row) in zip(cases, all_data)
        bins = StatUtils.mkcutoffs(amin, amax, nbuckets)

        hist = StatUtils.mkhist(amin, amax, row, nbuckets)
        hist = Float64.(hist)
        hist ./= Float64(nsamples)

        histplot = plt.bar(bins, hist, ylims=(0.0, 1.0))
        if !isnothing(mktitle)
            title = mktitle(param)
            plt.title!(histplot,title)
        end

        maxbin = max(maxbin, maximum(hist))

        push!(plots, histplot)
    end

    if rescale
        for plot in plots
            plt.ylims!(plot, (0.0, maxbin))
        end
    end

    return plots
end #plot_histograms

function make_cost_plot(
    cases,
    results,
    ;
    title=nothing,
    ylabel="cost",
    xlabel=nothing,
)
    cost_plot = plt.Plot()
    plt.plot!(cost_plot, cases, results;
        markershape=:circle,
        legend=false,
    )

    if !isnothing(title)
        plt.title!(cost_plot, title)
    end

    plt.ylabel!(ylabel)

    if !isnothing(xlabel)
        plt.xlabel!(xlabel)
    end

    return cost_plot
end #make_cost_plot

function get_cost_bounds(inst::LotSizingInstance, xs)
    mmin = OptModel.opt_scenario_cost(inst,xs)
    JuMP.optimize!(mmin)

    mmax = PesModel.pes_scenario_cost(inst, xs)
    JuMP.optimize!(mmax)

    return JuMP.objective_value.((mmin,mmax))
end #get_cost_bounds

function rejection_sample_arrivals(
    inst::LotSizingInstance,
    N::Int,
    ;
    seed=nothing,
)
    rng = Random.Xoshiro(seed)

    (;T) = inst
    samples = [zeros(Int, T) for _ in 1:N]

    feasible_arrivals = [feasible_arrival_times(inst, t) for t in 1:inst.T]

    for i in 1:N
        while true
            for t in 1:T
                samples[i][t] = Random.rand(rng, feasible_arrivals[t])
            end

            if Instance.valid_arrival(inst, samples[i], true)
                break
            end
        end
    end

    return samples
end #rejection_sample_arrivals

function layout_title(title)
    return plt.plot(title=title,
        framestyle=nothing,
        showaxis=false,
        xticks=false,yticks=false,
        margin=0mm
    )
end #layout_title

end #ExperimentUtils
