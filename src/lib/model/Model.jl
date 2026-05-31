__precompile__(true)
module Model

import ..MinMinModel
create_minmin_model = MinMinModel.create_minmin_model
export create_minmin_model

import ..MinMaxModel
create_pes_model = MinMaxModel.create_pes_model
export create_pes_model

import ..RModel
create_R_model = RModel.create_R_model
export create_R_model

end # module
