module CovidDepressionAnalysis

using DataFrames
# include packages
include("constants.jl")
include("data_processing.jl")

# Export functions
export extract_wave, combine_wave_data

# Export const variables
export DEMOGRAPHICS, PSYCHOLOGICAL, EXTERNAL, QUALITY, EXTRAS, VARNAME_MAPPING

end
