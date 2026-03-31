__precompile__(true)
module StatUtils

using DelimitedFiles
using StatsBase
using LinearAlgebra

export mkcutoffs, mkhist

"""
Create cutoffs for `N` bins spread uniformly between `vmin` and `vmax`.
"""
function mkcutoffs(vmin, vmax, N)
    return LinRange(vmin, vmax, N+1)
end

"""
Divide the list of `values` into `N` evenly spaced bins between `vmin` and `vmax`.

Arguments:
* `vmin` - minimum value in `values`
* `vmax` - maximum value in `values`
* `values` - array of values
* `N` - number of bins

Returns: array containing histogram of occurences of values in the bins.
"""
function mkhist(vmin, vmax, values, N)
    bins = mkcutoffs(vmin,vmax, N)
    hist = fit(Histogram, values, bins)

    return hist.weights
end

"""
Perform computation returning an array with file caching. If file is present return the
cached value, otherwise execute the computation.

Arguments:
* compute - function perfoming computation,
* filename - path to the desired file.

Optional arguments:
* force - ignore the cache file and always perform the computaiton,
* dlm - delimiter in the cache file,
* typ - type of the values.

Returns: array of values.
"""
function with_file_cache(compute, filename; force=false, dlm=',', typ=Float64)
    if !force && isfile(filename)
        return readdlm(filename, dlm, typ)
    end

    A = compute()

    writedlm(filename, A, dlm)

    return A
end

end #StatUtils
