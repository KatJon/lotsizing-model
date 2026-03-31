__precompile__(true)
module Model

import ..OptModel
create_opt_model = OptModel.create_opt_model
export create_opt_model

import ..PesModel
create_pes_model = PesModel.create_pes_model
export create_pes_model

import ..RModel
create_R_model = RModel.create_R_model
export create_R_model

end # module
