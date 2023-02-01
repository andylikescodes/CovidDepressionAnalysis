# The script to output data wave separately. 
using Revise
push!(LOAD_PATH, "../src") 
using DataFrames
using CovidDepressionAnalysis
using CSV

# Define helper functions
function process_target_waves()
    # Load Data
    data = CSV.read("data/data_all_impute.csv", DataFrame)

    # interpolate age from wave 1 to other waves
    interpolate_age_race(data)
    # Load Data
    external_data = CSV.read("data/externalMeasures_county.csv", DataFrame)
    external_subset = external_data[external_data[!, "wave_day"] .== "start_date", :]
    total_data = leftjoin(data, external_subset, on=["PROLIFIC_PID", "wave"])
    
    state_census = CSV.read("data/NST-EST2021-alldata.csv", DataFrame)
    state_census[!, :state] = lowercase.(filter.(isascii, state_census[!, :NAME]))

    total_data = leftjoin(total_data, state_census, on=:state)

    # Processing for wave6, 8, 10, 12, 14, 16
    target_waves = [1, 6, 8, 10, 12, 14, 16]

    # Variables selected
    selected_variables = [DEMOGRAPHICS; PSYCHOLOGICAL; EXTERNAL; QUALITY; EXTRAS]

    # Extract data
    for w in target_waves
        waveName = "w"*string(w)
        w_data = rename!(extract_wave(total_data, w, selected_variables), VARNAME_MAPPING)
        # Binarize Gender and Race
        w_data[!, "Gender"] = w_data[!, "Gender"] .== 1
        w_data[!, "Race_AA"] = w_data[!, "Race"] .== 4
        w_data[!, "Race_W"] = w_data[!, "Race"] .== 5

        CSV.write("output/v2/"*waveName*".csv", w_data)
    end
end

process_target_waves()

