using Revise
push!(LOAD_PATH, "../src") 
using DataFrames
using CovidDepressionAnalysis
using CSV
using Counterfactuals
using Plots
using PairPlots

function corner_plots(data_path, save_path)
    println("processing:")
    println(data_path)
    data = CSV.read(data_path, DataFrame)
    
    variable_of_interest_1 = ["Gender", "Education", "Income", "Political_Views", "Age", "Race", "Depression", "Depression_adj", "Stress", "Anxiety", "Loneliness", "Emotional_Support"]
    variable_of_interest_2 = ["Depression", "Depression_adj", "Mandatory_SAH", "GatheringStrictness", "stayHome_order_code", "gatherBan_order_code", "slope_new_cases", "slope_new_deaths", "cvd_cases_7d_avg", "cvd_deaths_7d_avg", "State_Population"]
    
    savefig(corner(data[Not(isnan.(data[!, :cvd_cases_7d_avg])), variable_of_interest_1], plotcontours=false, filterscatter=false), save_path*".svg")
    savefig(corner(data[Not(isnan.(data[!, :cvd_cases_7d_avg])), variable_of_interest_2], plotcontours=false, filterscatter=false), save_path*"2.svg")

end

corner_plots("output/IP_weighted_sources_pop/w6.csv", "output/figures/w6_corner")
corner_plots("output/IP_weighted_sources_pop/w8.csv", "output/figures/w8_corner")
corner_plots("output/IP_weighted_sources_pop/w10.csv", "output/figures/w10_corner")
corner_plots("output/IP_weighted_sources_pop/w12.csv", "output/figures/w12_corner")
corner_plots("output/IP_weighted_sources_pop/w14.csv", "output/figures/w14_corner")
corner_plots("output/IP_weighted_sources_pop/w16.csv", "output/figures/w16_corner")