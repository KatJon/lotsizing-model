__precompile__(true)
module Model

import ..MinMinModel
create_minmin_model = MinMinModel.create_minmin_model
export create_minmin_model

import ..MinMaxModel
create_minmax_model = MinMaxModel.create_minmax_model
export create_minmax_model

import ..RModel
create_R_model = RModel.create_R_model
export create_R_model

end # module
