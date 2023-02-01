# The script to output data wave separately. 
using Revise
push!(LOAD_PATH, "../src") 
using DataFrames
using CovidDepressionAnalysis
using CSV

# define extract function
function fur_tetrad()
    # Processing for wave6, 8, 10, 12, 14, 16
    target_waves = [6, 8, 10, 12, 14, 16]
    for w in target_waves
        waveName = "w"*string(w)
        data = CSV.read("output/v2/"*waveName*".csv", DataFrame)
        data = data[!, TETRAD_VARS]
        data[!,:Mandatory_SAH] = data[!,:Mandatory_SAH] .== 1
        CSV.write("output/Tetrad_output/"*waveName*".csv", data)
    end
end

fur_tetrad()