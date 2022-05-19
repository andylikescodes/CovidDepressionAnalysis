using Revise
push!(LOAD_PATH, "../src") 
using DataFrames
using CovidDepressionAnalysis
using CSV
using Counterfactuals
using Plots
using InvertedIndices
using Statistics

function IPW_adj_dropout_pop(w_data_path, save_path)
    println("Processing")
    println(w_data_path)
    w1 = CSV.read("output/v2/w1.csv", DataFrame)
    w_data = CSV.read(w_data_path, DataFrame)

    w1[!, :state_pop_zscore] = (w1[!, :State_Population] .- mean(w1[!, :State_Population])) ./ std(w1[!, :State_Population])
    w_data[!, :state_pop_zscore] = (w_data[!, :State_Population] .- mean(w_data[!, :State_Population])) ./ std(w_data[!, :State_Population])


    # First IP weighting adjust for low quality dropouts
    adjustments = ["Income", "Gender", "Age", "Education", "Race_AA", "Race_W","state_pop_zscore", "Mandatory_SAH"]
    square_terms = ["Income", "Age", "Education", "state_pop_zscore"]
    treatment = "low_quality"
    target = "Depression"

    w_data[:, :low_quality] = sum.(eachrow(w_data[!, QUALITY_OUTPUT_NAME])) .>= 1

    fit_low_quality = cal_ipw(w_data, adjustments, square_terms, treatment)

    # Second IP weighting adjust for dropouts based on the demographc data on the first wave
    combined = combine_wave_data(w1, w_data, "PROLIFIC_PID", "w1", "this")

    combined[!, :dropout] = combined[!, target*"_this"].===missing

    adjustments = ["Income_w1", "Gender_w1", "Age_w1", "Education_w1", "Race_AA_w1", "Race_W_w1", "state_pop_zscore_w1", "Mandatory_SAH_w1"]
    square_terms = ["Income_w1", "Age_w1", "Education_w1", "state_pop_zscore_w1"]
    treatment = "dropout"
    target = "Depression"

    fit_dropout = cal_ipw(combined, adjustments, square_terms, treatment)

    # combined probability
    combined[!, :total_w] = combined[!, :low_quality_w_this] .* combined[!, :dropout_w]

    # Depression score adjusted for low quality and dropouts
    combined[!, :Depression_adj] = combined[!, :Depression_this] .* (1 ./ combined[!, :total_w])

    selected = combined[(combined[!, :Depression_adj] .!==missing) .& (combined[!, :low_quality_this] .== 0), Not(r"_w1")]

    # Clean up the variable names
    original_name = names(selected)
    new_name = replace.(original_name, "_this" => "") 
    rename!(selected, Dict(zip(original_name, new_name)))

    # Save data
    CSV.write(save_path, selected)

    # Save model output

    println(fit_low_quality)
    println(fit_dropout)

    return fit_low_quality, fit_dropout
end

files = ["w6", "w8", "w10", "w12", "w14", "w16"]
for f in files
    IPW_adj_dropout_pop("output/v2/"*f*".csv", "output/IP_weighted_sources_pop/"*f*".csv")
end


