# Define important variables into the analysis
const DEMOGRAPHICS = ["PROLIFIC_PID", "Sex", "Education", "Income_HH", "DemM7_KNN", "prlfc_dem_age", "DemC9"]
const PSYCHOLOGICAL = ["BDI_total_raw", "PSS_Total", "STAI_State_raw", "Fear_COVID_raw_KNN", "NIH_TB_Emot_Support_raw_total_KNN", "NIH_TB_Loneliness_raw_total_KNN"]
const EXTERNAL = [ "slope_new_cases", "slope_new_deaths", "GatheringStrictness_KNN", "Mandatory_business_closure", "Mandatory_PPE_masks", "gatherBan_order_code", "Mandatory_SAH", "stayHome_order_code", "cvd_cases_7d_avg", "cvd_deaths_7d_avg"]
const QUALITY = ["more_than_1_attQ_failed", "string_outlier_core", "response_consistency"]

# Some extra variables that are needed for data extraction
# RW1_8: people who stay at home all the time
const EXTRAS = ["wave", "RW1_8"]

# Output variable names mapping
const DEMOGRAPHICS_OUTPUT_NAME = ["PROLIFIC_PID", "Gender", "Education", "Income", "Political_Views", "Age", "Race"]
const PSYCHOLOGICAL_OUTPUT_NAME = ["Depression", "Stress", "Anxiety", "Fear_COVID", "Emotional_Support", "Loneliness"]
const EXTERNAL_OUTPUT_NAME =  ["slope_new_cases", "slope_new_deaths", "GatheringStrictness", "Mandatory_business_closure", "Mandatory_PPE_masks", "gatherBan_order_code", "Mandatory_SAH", "stayHome_order_code", "cvd_cases_7d_avg", "cvd_deaths_7d_avg"]
const QUALITY_OUTPUT_NAME = ["more_than_1_attQ_failed", "string_outlier_core", "response_consistency"]

tmp1 = vcat(DEMOGRAPHICS, PSYCHOLOGICAL, EXTERNAL, QUALITY)
tmp2 = vcat(DEMOGRAPHICS_OUTPUT_NAME, PSYCHOLOGICAL_OUTPUT_NAME, EXTERNAL_OUTPUT_NAME, QUALITY_OUTPUT_NAME)
const VARNAME_MAPPING = Dict(zip(tmp1, tmp2))

