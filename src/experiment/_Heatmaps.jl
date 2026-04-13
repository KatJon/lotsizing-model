import LotsizingModel as LM

import .LM.StatUtils

using .LM.Instance
using .LM.ExperimentUtils

import Plots as plt

using Accessors: @set,@reset
using Plots.PlotMeasures
using ProgressMeter
using JuMP
using DelimitedFiles
using Printf
using Statistics

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

function hist_hurwicz(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
    force=false,
)
    hpoints=50
    gammas=0:inst.T

    alphas = LinRange(0.0, 1.0, hpoints)

    mkpath("output")
    all_costs=zeros(Float64, length(gammas), length(alphas), N)
    plots=[]

    samples = sample_arrivals(inst, N; ignore_budget=true)

    println("Simulations.")
    @showprogress for (i,gamma) in enumerate(gammas)
        tmpname = isnothing(name) ? "TMP" : name
        SAVEFILE="output/$(tmpname)_sampleHG1_$(gamma)_$(hpoints)_$(N).csv"

        inst1 = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma1, gamma)

        results = StatUtils.with_file_cache(SAVEFILE; force=force) do 
            costs = zeros(Float64, length(alphas), length(samples))

            for (j,alpha) in enumerate(alphas)
                model,x,y = LM.Hurwicz.create_hurwicz_model(inst1, alpha)

                optimize!(model)

                xs = JuMP.value.(x)
                ys = JuMP.value.(y)

                for (k, sample) in enumerate(samples)
                    costs[j,k] = prod_lead_cost(inst,xs,ys,sample)
                end
            end

            return costs
        end

        all_costs[i,:,:] = results
    end

    buckets = 1 + 2*ceil(Int64, log2(N))
    vmin = minimum(all_costs[i] for i in eachindex(all_costs))
    vmax = maximum(all_costs[i] for i in eachindex(all_costs))
    cutoffs = StatUtils.mkcutoffs(vmin, vmax, buckets)

    @showprogress for (i,gamma) in enumerate(gammas)
        hist = zeros(Float64, length(alphas), buckets)

        for (j,_) in enumerate(alphas)
            hist[j,:] = StatUtils.mkhist(vmin, vmax, all_costs[i,j,:], buckets)
        end

        hist ./= N

        cont = plt.contourf(cutoffs[begin:end-1], alphas, hist,
            levels=LinRange(0.0, 1.0, 100),
            color=plt.cgrad(:turbo, scale=:exp), lw=0,
            bottom_margin=10mm
        )
        plt.title!(cont, "Sim. Hurwicz model (\$\\Gamma_1=$(gamma)\$)")
        plt.xlabel!(cont, "cost")
        plt.ylabel!(cont, "\$\\alpha\$")

        push!(plots, cont)
    end

    allplots = plt.plot(plots...; layout=length(gammas), size=(3000,2000))

    if isnothing(name)
        plt.gui(allplots)
    else
        plt.savefig(allplots, "fig/$(name)_HG1_heatmap.png")
    end
end

function sampling_R_Gamma1(
    inst::LotSizingInstance,
    ;
    N=100,
    name=nothing,
    force=false,
)
    Rpoints=50
    gammas=0:inst.T

    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    opt_bound = opt_value(inst_interval)
    pes_bound = pes_value(inst_interval)

    rbudgets = LinRange(opt_bound, 1.5*pes_bound, Rpoints)

    mkpath("output")
    all_costs=zeros(Float64, length(gammas), length(rbudgets), N)
    plots=[]

    samples = sample_arrivals(inst, N; ignore_budget=true)

    println("Simulations.")
    @showprogress for (i,gamma) in enumerate(gammas)
        tmpname = isnothing(name) ? "TMP" : name
        SAVEFILE="output/$(tmpname)_sampleRG1_$(gamma)_$(Rpoints)_$(N).csv"

        inst1 = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Gamma1, gamma)

        results = StatUtils.with_file_cache(SAVEFILE; force=force) do 
            costs = zeros(Float64, length(rbudgets), length(samples))

            for (j,rbound) in enumerate(rbudgets)
                inst2 = @set inst1.rbound=rbound
                _,x,y = solveR(inst2)

                xs = JuMP.value.(x)
                ys = JuMP.value.(y)

                for (k, sample) in enumerate(samples)
                    costs[j,k] = prod_lead_cost(inst,xs,ys,sample)
                end
            end

            return costs
        end

        all_costs[i,:,:] = results
    end

    buckets = 1 + 2*ceil(Int64, log2(N))
    vmin = minimum(all_costs[i] for i in eachindex(all_costs))
    vmax = maximum(all_costs[i] for i in eachindex(all_costs))
    cutoffs = StatUtils.mkcutoffs(vmin, vmax, buckets)

    @showprogress for (i,gamma) in enumerate(gammas)
        hist = zeros(Float64, length(rbudgets), buckets)

        for (j,_) in enumerate(rbudgets)
            hist[j,:] = StatUtils.mkhist(vmin, vmax, all_costs[i,j,:], buckets)
        end

        hist ./= N

        cont = plt.contourf(cutoffs[begin:end-1], rbudgets, hist,
            levels=LinRange(0.0, 1.0, 100),
            color=plt.cgrad(:turbo, scale=:exp), lw=0,
            bottom_margin=10mm
        )
        plt.plot!(cont, rbudgets, rbudgets, color=:red)
        plt.title!(cont, "Sim. R* model (\$\\Gamma_1=$(gamma)\$)")
        plt.xlabel!(cont, "cost")
        plt.ylabel!(cont, "R* budget")

        push!(plots, cont)
    end

    allplots = plt.plot(plots...; layout=length(gammas), size=(3000,2000))

    if isnothing(name)
        plt.gui(allplots)
    else
        plt.savefig(allplots, "fig/$(name)_RG1_heatmap.png")
    end
end

function contourR(
    inst::LotSizingInstance
    ;
    name=nothing,
    force=false
)
    Rpoints=30

    budgets=1:inst.T

    inst_interval = @set inst.uncertainty=Uncertainty.Model(Uncertainty.Interval, nothing)
    opt_bound = opt_value(inst_interval)
    pes_bound = pes_value(inst_interval)

    rbudgets = LinRange(0.9*pes_bound, 1.5*pes_bound, Rpoints)

    tmpname = isnothing(name) ? "TMP" : name
    SAVEFILE="output/$(tmpname)_rgamma1exp_$(Rpoints).csv"

    mkpath("output")
    results = StatUtils.with_file_cache(SAVEFILE; force=force) do 
        res = zeros(Float64, length(budgets), length(rbudgets))

        @showprogress for ((i,gamma),(j,rbound)) in Iterators.product(
            enumerate(budgets),
            enumerate(rbudgets),
        )
            cost, = solveR(inst, Gamma1=gamma, rbound=rbound)
            res[i,j] = cost
        end

        return res
    end

    cont = plt.contourf(rbudgets, budgets, results, levels=20, color=:turbo, lw=0)
    plt.title!(cont, "R* cost")
    plt.xlabel!(cont, "R* budget")
    plt.ylabel!(cont, "\$\\Gamma_1\$")

    if isnothing(name)
        plt.gui(cont)
    else
        plt.savefig("fig/$(name)_Rsurf.png")
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

