# The core scripts for the analysis
using Revise
push!(LOAD_PATH, "../src") 
using DataFrames
using CovidDepressionAnalysis
using CSV

# Define helper functions
function process_target_waves()
    # Load Data
    data = CSV.read("data/data_all_impute.csv", DataFrame)
    external_data = CSV.read("../data/externalMeasures_county.csv", DataFrame)
    external_subset = external_data[external_data[!, "wave_day"] .== "start_date", :]
    total_data = leftjoin(data, external_subset, on=["PROLIFIC_PID", "wave"])

    # Processing for wave6, 8, 10, 12, 14, 16
    target_waves = [6, 8, 12, 14, 16]

    # Variables selected
    selected_variables = [DEMOGRAPHICS; PSYCHOLOGICAL; EXTERNAL; QUALITY; EXTRAS]

    # Extract data
    w1 = rename!(extract_wave(total_data, 1, selected_variables), VARNAME_MAPPING)
    for w in target_waves
        waveName = "w"*string(w)
        w_data = rename!(extract_wave(total_data, w, selected_variables), VARNAME_MAPPING)
        combine_data = combine_wave_data(w1, w_data, "PROLIFIC_PID", "w1", waveName)
        CSV.write("output/v1/"*waveName*".csv", combine_data)
    end
end

process_target_waves()

