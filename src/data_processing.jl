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

# Modify variable names
