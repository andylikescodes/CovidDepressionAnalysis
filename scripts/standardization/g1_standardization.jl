# The core scripts for the analysis
using Revise
push!(LOAD_PATH, "../src") 
using DataFrames
using Counterfactuals
using CSV
using Statistics
using StatsBase

function standardization_estimation(data_path, bootstrap)
    println("Processing:")
    println(data_path)
    data = CSV.read(data_path, DataFrame)
    data = data[Not(isnan.(data[!, :cvd_cases_7d_avg])), :]

    adjustments = ["cvd_cases_7d_avg", "cvd_deaths_7d_avg", "Gender", "Age", "Education", "Race_AA", "Race_W", "Political_Views"]
    square_terms = ["cvd_cases_7d_avg", "cvd_deaths_7d_avg", "Age", "Education", "Political_Views"]
    treatment = "Mandatory_SAH"
    target = "Depression_adj"

    effects = estimate_standardization(data, adjustments, square_terms, treatment, target, bootstrap)
    Non_NA_vec = effects[.!isnan.(effects)]
    mean_value = mean(Non_NA_vec)
    CI = [percentile(Non_NA_vec, 2.5), percentile(Non_NA_vec, 97.5)]
    println("Estimated Effects:")
    println(mean_value)
    println(CI)
    #return effects, mean, CI 
end

files = ["w6", "w8", "w10", "w12", "w14", "w16"]
bootstrap = 1000
for f in files
    standardization_estimation("output/IP_weighted_sources/"*f*".csv", bootstrap)
end
