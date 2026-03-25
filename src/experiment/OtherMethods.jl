import LotsizingModel as LM
using .LM.Instance

import .LM.StatUtils

import Plots as plt
using Plots.PlotMeasures
using Accessors: @set,@reset
using ProgressMeter
using JuMP
using DelimitedFiles

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

function run()
    
end
