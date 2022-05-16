# Stores general processing function on a julia dataframe that are transferable to other projects
# A fucntion to create suffix or prefix for a dataframe
function create_suffix_vector_mapping(vec, suffix; except=[])
    if !isempty(except)
        deleteat!(vec, occursin.(vec, except))
    end
    new_vec = [join([x, "_", suffix]) for x in vec]
    Dict(zip(vec, new_vec))
end

function create_suffix_df(df, suffix; except=[])
    old_names = names(df)

    rename(df, create_suffix_vector_mapping(old_names, suffix; except=except))
end


