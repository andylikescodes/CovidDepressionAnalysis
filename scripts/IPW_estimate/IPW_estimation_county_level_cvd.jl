# The core scripts for the analysis
using Revise
push!(LOAD_PATH, "../src") 
using DataFrames
using Counterfactuals
using CSV
using Statistics
using StatsBase

function IPW_estimation(data_path, bootstrap, threshold)
    println("Processing:")
    println(data_path)
    data = CSV.read(data_path, DataFrame)
    data[!,:Gathering_Strictness_bin] = data[!, :GatheringStrictness] .<= threshold
    data = data[Not(isnan.(data[!, :cvd_cases_7d_avg])), :]

    adjustments = ["cvd_cases_7d_avg", "cvd_deaths_7d_avg"]
    square_terms = ["cvd_cases_7d_avg", "cvd_deaths_7d_avg"]
    treatment = "Gathering_Strictness_bin"
    target = "Depression_adj"

    effects, mean, CI = estimate_ipw(data, adjustments, square_terms, treatment, target, bootstrap)
    println("Estimated Effects:")
    println(mean)
    println(CI)
    #return effects, mean, CI 
end

IPW_estimation("output/IP_weighted_sources/w6.csv", 5000, 3)
IPW_estimation("output/IP_weighted_sources/w8.csv", 5000, 3)
IPW_estimation("output/IP_weighted_sources/w10.csv", 5000, 3)
IPW_estimation("output/IP_weighted_sources/w12.csv", 5000, 3)
IPW_estimation("output/IP_weighted_sources/w14.csv", 5000, 3)
IPW_estimation("output/IP_weighted_sources/w16.csv", 5000, 3)