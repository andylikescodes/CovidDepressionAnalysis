library('tidyverse')

data <- read_csv("data/Wave1-16_paper_release.csv")
external <- read_csv("data/w1-w16_external_state.csv")

# States mapping
states <- c("Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming");
states_encoded <- 1:50;
df <- data.frame(States=states, DemW3=states_encoded)

external_state_encoded <- external %>% right_join(df, by='States', keep=FALSE, unmatched="drop")
external_state_encoded$wave <- as.character(external_state_encoded$wave)

demographics <- c("CVDID", "DemM7", "prlfc_dem_age", "DemC9")
psychological <- c("BDI_total_raw", "PSS_Total", "STAI_State_raw", "Fear_COVID_raw", "NIH_TB_Emot_Support_raw_total", "NIH_TB_Loneliness_raw_total")
others <- c("wave", "DemW3", "States")

vars <- c(demographics, psychological, others)

#data$wave <- as.double(data$wave)

core_samples <- data %>% 
                    left_join(external_state_encoded, by=c("DemW3", "wave")) %>%
                    filter(sample == "core") %>%
                    select(all_of(vars))
                    
core_samples %>% filter(wave == "6") %>%
                    select(all_of(c("DemW3", "States", "CVDID")))               


