# Include functions from other scripts
include("utils.jl")
# Function to extract a specific wave of data
# wrapper function
function extract_wave(data, w, selected_variables)
    data[data[!, "wave"].==w, selected_variables]
end

# Combine wave1 data and other waves into a workable 
#function combine
function combine_wave_data(df1, df2, on, lsuffix, rsuffix)
    df1 = create_suffix_df(df1, lsuffix, except=[on])
    df2 = create_suffix_df(df2, rsuffix, except=[on])
    outerjoin(df1, df2, on=on)
end

# interpolate Age and Race for all waves
function interpolate_age_race(data)
    w1 = data[data[!, :wave].==1, :]
    unique_pids = unique(w1[!, :PROLIFIC_PID])
    for id in unique_pids
        id_idx = data[!, :PROLIFIC_PID].==id
        val_idx = w1[!, :PROLIFIC_PID].==id
        data[id_idx, :prlfc_dem_age] .= w1[val_idx, :prlfc_dem_age] 
        data[id_idx, :DemC9] .= w1[val_idx, :DemC9]  
    end 
end

# Process data for Tetrad
function fur_tetrad(data)
    data[!, TETRAD_VARS]
end

