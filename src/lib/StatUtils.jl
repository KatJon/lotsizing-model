module StatUtils

using DelimitedFiles
using StatsBase
using LinearAlgebra

export mkcutoffs, mkhist

function mkcutoffs(vmin, vmax, N)
    return LinRange(vmin, vmax, N+1)
end

function mkhist(vmin, vmax, values, N)
    bins = mkcutoffs(vmin,vmax, N)
    hist = fit(Histogram, values, bins)

    return hist.weights
end

function with_file_cache(compute, filename; force=false, dlm=',', typ=Float64)
    if !force && isfile(filename)
        return readdlm(filename, dlm, typ)
    end

    A = compute()

    writedlm(filename, A, dlm)

    return A
end

end #StatUtils
