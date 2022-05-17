module CovidDepressionAnalysis

using DataFrames
# include packages
include("constants.jl")
include("data_processing.jl")

# Export functions
export extract_wave, combine_wave_data, interpolate_age_race

# Export const variables
export DEMOGRAPHICS, PSYCHOLOGICAL, EXTERNAL, QUALITY, EXTRAS
export DEMOGRAPHICS_OUTPUT_NAME, PSYCHOLOGICAL_OUTPUT_NAME, EXTERNAL_OUTPUT_NAME, QUALITY_OUTPUT_NAME, EXTRAS, VARNAME_MAPPING

end
