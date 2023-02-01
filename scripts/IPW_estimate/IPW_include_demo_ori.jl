# The core scripts for the analysis
using Revise
push!(LOAD_PATH, "../src") 
using DataFrames
using Counterfactuals
using CSV
using Statistics
using StatsBase

function IPW_estimation(data_path, bootstrap)
    println("Processing:")
    println(data_path)
    data = CSV.read(data_path, DataFrame)
    data = data[Not(isnan.(data[!, :cvd_cases_7d_avg])), :]

    adjustments = ["cvd_cases_7d_avg", "cvd_deaths_7d_avg", "Gender", "Age", "Education", "Race_AA", "Race_W", "Political_Views"]
    square_terms = ["cvd_cases_7d_avg", "cvd_deaths_7d_avg", "Age", "Education", "Political_Views"]
    treatment = "Mandatory_SAH"
    target = "Depression"

    effects = estimate_ipw(data, adjustments, square_terms, treatment, target, bootstrap)
    Non_NA_vec = effects[.!isnan.(effects)]
    mean_value = mean(Non_NA_vec)
    CI = [percentile(Non_NA_vec, 2.5), percentile(Non_NA_vec, 97.5)]
    println("Estimated Effects:")
    println(mean_value)
    println(CI)
    #return effects, mean, CI 
end

IPW_estimation("output/v2/w6.csv", 100)
IPW_estimation("output/v2/w8.csv", 100)
IPW_estimation("output/v2/w10.csv", 100)
IPW_estimation("output/v2/w12.csv", 100)
IPW_estimation("output/v2/w14.csv", 100)
IPW_estimation("output/v2/w16.csv", 100)