import LotsizingModel as LM

using JuMP

module DM
    include("check/DirectModel.jl")
end

inst = LM.load_instance("data/simpl.dat")
