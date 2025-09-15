__precompile__(true)
module Parse

using ..Instance

function get_next_line(file::IOStream)::String
    while true
        line = readline(file)
        if !startswith(line, "#")
            return line
        end
    end
end #get_next_line

function parse_line(
    type::DataType,
    line::String;
    sep::String = " ",
)::Vector{type}
    arr = split(line, sep)

    return [parse(type, x) for x in arr]
end #parse_line

function assert_array_length(
    array::AbstractVector,
    expected_length::Int,
    error_message::String,
)::Nothing
    if length(array) != expected_length
        throw(error_message)
    end
end #assert_array_length

UNCERTAINTY_TYPES = ["NONE", "INTERVAL", "GAMMA1", "GAMMA2"]

function parse_uncertainty(
    line::String;
    sep::String = " ",
)::Uncertainty.Model
    data = split(line, sep)

    if length(data) == 0
        throw("Missing uncertainty model specification")
    end

    type = data[1]

    if !(type in UNCERTAINTY_TYPES)
        throw("Unknown uncertatiny type $(type)")
    end

    if type == "NONE"
        return Uncertainty.Model(Uncertainty.None, nothing)
    end

    if type == "INTERVAL"
        return Uncertainty.Model(Uncertainty.Interval, nothing)
    end
    
    if length(data) < 2
        throw("Missing uncertainty parameter")
    end

    gamma = parse(Int, data[2])

    if type == "GAMMA1"
        return Uncertainty.Model(Uncertainty.Gamma1, gamma)
    end

    if type == "GAMMA2"
        return Uncertainty.Model(Uncertainty.Gamma2, gamma)
    end

    # Should be unreachable
    throw("Unexpected error in parsing uncertainty model.")
end #parse_uncertainty

RBOUND_TYPES = ["NONE", "R*"]

function parse_rbound(
    line::String;
    sep::String=" ",
)::Union{Nothing, Float64}
    data = split(line, sep)

    if length(data) == 0
        throw("Missing R* bound specification")
    end

    type = data[1]

    if !(type in RBOUND_TYPES)
        throw("Unknown R* bound type $(type)")
    end

    if type == "NONE"
        return nothing
    end

    if length(data) < 2
        throw("Missing R* bound value")
    end

    if type == "R*"
        bound = parse(Float64, data[2])

        return bound
    end

    throw("Unexpected error parsing R* bound")
end #parse_rbound

function parse_instance(
    file::IOStream,
)::LotSizingInstance
    # time periods
    line = get_next_line(file)
    (T, Tplus) = parse_line(Int64, line)

    # production period parameters
    line = get_next_line(file)
    setup_cost = parse_line(Float64, line)
    assert_array_length(setup_cost, T,
        "Wrong number of setup cost coefficients") 

    line = get_next_line(file)
    production_cost = parse_line(Float64, line)
    assert_array_length(production_cost, T,
        "Wrong number of production cost coefficients")

    line = get_next_line(file)
    production_capacity = parse_line(Float64, line)
    assert_array_length(production_capacity, T,
        "Wrong number of production capacity coefficients")

    # lead time characterization
    line = get_next_line(file)
    (common_lead_time,) = parse_line(Int64, line)
    
    line = get_next_line(file)
    late_deviation = parse_line(Int64, line)
    assert_array_length(late_deviation, T,
        "Wrong number of late delivery deviations") 

    line = get_next_line(file)
    early_deviation = parse_line(Int64, line)
    assert_array_length(early_deviation, T,
        "Wrong number of late delivery deviations")
    
    # future periods
    line = get_next_line(file)
    demand = parse_line(Float64, line)
    assert_array_length(demand, Tplus - T,
        "Wrong number of demand coefficients")

    line = get_next_line(file)
    inventory_cost = parse_line(Float64, line)
    assert_array_length(inventory_cost, Tplus - T,
        "Wrong number of inventory cost coefficients")

    line = get_next_line(file)
    backordering_cost = parse_line(Float64, line)
    assert_array_length(backordering_cost, Tplus - T,
        "Wrong number of backordering cost coefficients")

    # uncertainty modelling
    line = get_next_line(file)
    uncertainty = parse_uncertainty(line)

    # R* bound
    line = get_next_line(file)
    rbound = parse_rbound(line)

    return LotSizingInstance(
        # time periods
        T, Tplus,
        # production period parameters
        setup_cost,
        production_cost,
        production_capacity,
        # lead time characterization
        common_lead_time,
        late_deviation,
        early_deviation,
        # future periods
        demand,
        inventory_cost,
        backordering_cost,
        # uncertainty modelling
        uncertainty,
        # R* bound,
        rbound,
    )
end #parse_instance

end #Parse
