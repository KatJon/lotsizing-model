import LotsizingModel as LM

module DM
    include("check/DirectModel.jl")
end

inst = LM.load_instance("data/simpl.dat")
