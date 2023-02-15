library('tidyverse')

lookup_states <- function(data){
    wave_1 <- data %>% filter(wave=="1")
    wave_1[wave_1$DemC23 == 10, "DemC23"] <- NA
    for (i in 1:dim(data)[1]){
        if (is.na(data[i, "DemW3"])){
            data[i, "DemW3"] = wave_1[wave_1$CVDID==data[i, "CVDID"]$CVDID, "DemW3"]
        }
        data[i, "DemC23"] = wave_1[wave_1$CVDID==data[i, "CVDID"]$CVDID, "DemC23"]
        data[i, "prlfc_dem_age"] = wave_1[wave_1$CVDID==data[i, "CVDID"]$CVDID, "prlfc_dem_age"]
        data[i, "DemC9"] = wave_1[wave_1$CVDID==data[i, "CVDID"]$CVDID, "DemC9"]
        data[i, "DemC5"] = wave_1[wave_1$CVDID==data[i, "CVDID"]$CVDID, "DemC5"]
    }
    return (data)
}

data <- read_csv("data/Wave1-16_paper_release.csv")
external_state <- read_csv("data/w1-w16_external_state.csv")
external_county <- read_csv("data/externalMeasures_county.csv")

ID_mapping <- read_csv("data/PROLIFIC_PID_2_CVDID.csv")

external_county_selected <- external_county %>% left_join(ID_mapping, by="PROLIFIC_PID", unmatched="drop") %>%
                    filter(wave_day=="start_date") %>%
                    select(c("CVDID", "wave", "wave_day", "cvd_cases_7d_avg", "cvd_deaths_7d_avg", "cvd_cases_7d_avg_per_100k", "cvd_deaths_7d_avg_per_100k"))


wave_1 <- data %>% filter(wave=="1")
wave_1[wave_1$DemC23 == 10, "DemC23"]

# States mapping
states <- c("Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming");
states_encoded <- 1:50;
df <- data.frame(States=states, DemW3=states_encoded)

# Use wave 1 data and impute
data = lookup_states(data)

external_state_encoded <- external_state %>% right_join(df, by='States', keep=FALSE, unmatched="drop")
external_state_encoded$wave <- as.character(external_state_encoded$wave)
external_county_selected$wave <- as.character(external_county_selected$wave)

ID <- c("CVDID")
demographics <- c("DemM7", "prlfc_dem_age", "DemC9", "DemC23", "DemC5")
psychological <- c("BDI_total_raw", "PSS_Total", "STAI_State_raw", "Fear_COVID_raw", "NIH_TB_Emot_Support_raw_total", "NIH_TB_Loneliness_raw_total")
others <- c("DemW3", "States", "Mandatory_SAH", "slope_new_cases", "slope_new_deaths", 'cvd_cases_7d_avg', "cvd_deaths_7d_avg", "cvd_cases_7d_avg_per_100k", "cvd_deaths_7d_avg_per_100k")
wave <- c("wave")


vars <- c(ID, demographics, psychological, others, wave)

#data$wave <- as.double(data$wave)

core_samples <- data %>% left_join(external_state_encoded, by=c("DemW3", "wave")) %>%
                            left_join(external_county_selected, by=c("CVDID", "wave"), unmatched="drop")

core_samples <- data %>% 
                    left_join(external_state_encoded, by=c("DemW3", "wave")) %>%
                    left_join(external_county_selected, by=c("CVDID", "wave"), unmatched="drop") %>%
                    filter(sample == "core") %>%
                    filter(wave != "15b") %>%
                    select(all_of(vars))

write.csv(core_samples, "data/selected_core_samples.csv")
                             

core_samples %>% select(all_of(c("DemM7", "prlfc_dem_age", "DemC9", "DemC23","DemC5", "wave"))) %>%
                    filter(wave=="2")

