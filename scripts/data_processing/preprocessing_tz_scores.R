library('tidyverse')

# A function to link the demographic variables with wave1 and all other waves
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

# Load the source data
data <- read_csv("data/Wave1-16_paper_release.csv")
# Use wave 1 data and impute
data <- lookup_states(data)

# mapping income
income_recode <- c("1"=1, "2"=1, "3"=1, "4"=2, "5"=3, "6"=4, "7"=5, "8"=6, "9"=7)
data <- data %>% mutate(Income = ifelse( wave %in% c(2,3,4,5,6), recode(DemW18_R2, !!!income_recode), 
                                ifelse(wave %in% 7:16, recode(DemW18.2, !!!income_recode),
                                    DemW18_R1))) %>%
                 mutate(Income = ifelse(Income == 7, NA, Income))

data <- data %>% mutate(Income_v2 = ifelse(wave %in% c(2,3,4,5,6), DemW18_R2,
                                    ifelse(wave %in% 7:16, DemW18.2, 
                                    NA))) %>%
                 mutate(Income_v2 = ifelse(Income_v2 == 9, NA, Income_v2))

# Load external data
external_state <- read_csv("data/w1-w16_external_state.csv")
external_county <- read_csv("data/externalMeasures_county.csv")

# Extra county data for lat/lng/pop from https://simplemaps.com/data/us-counties
external_county_extra <- read_csv("data/uscounties.csv") 

# Process the county data with lat/lng/pop
external_county_extra <- external_county_extra %>% mutate(county = tolower(county)) %>%
                                                    mutate(state_name = tolower(state_name)) %>%
                                                    rename(state = state_name) %>%
                                                    mutate(county=ifelse(state=='louisiana', paste(county, 'parish'), county), county=ifelse(state=='alaska', paste(county, 'borough'), county))

# Create mapping for external PROLIFIC_PID with CVDID
ID_mapping <- read_csv("data/PROLIFIC_PID_2_CVDID.csv")
external_county_selected <- external_county %>% left_join(ID_mapping, by="PROLIFIC_PID", unmatched="drop") %>%
                    filter(wave_day=="start_date") %>%
                    select(c("CVDID", "wave","county", "wave_day", "cases_avg", "deaths_avg", "cases_avg_per_100k", "deaths_avg_per_100k"))

# States mapping
states <- c("Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming");
states_encoded <- 1:50;
df <- data.frame(States=states, DemW3=states_encoded)

external_state_encoded <- external_state %>% right_join(df, by='States', keep=FALSE, unmatched="drop")
external_state_encoded$wave <- as.character(external_state_encoded$wave)
external_county_selected$wave <- as.character(external_county_selected$wave)

ID <- c("CVDID")
demographics <- c("DemM7", "prlfc_dem_age", "DemC9", "DemC23", "DemC5", "Income", "Income_v2")
psychological <- c("BDI_total_raw", "PSS_Total", "STAI_State_Tscore", "Fear_COVID_raw", "NIH_TB_Emot_Support_uncorrTscore", "NIH_TB_Loneliness_uncorrTscore", "NEO_N_z-score", "NEO_E_z-score", "NEO_O_z-score", "NEO_A_z-score", "NEO_C_z-score")
others <- c("DemW3", "state", 'county', "Mandatory_SAH", "slope_new_cases", "slope_new_deaths", "cases_avg", "deaths_avg", "cases_avg_per_100k", "deaths_avg_per_100k", "lat", 'lng', "population", "rake_weights")
wave <- c("wave")

vars <- c(ID, demographics, psychological, others, wave)

# Process and Select the data
core_samples <- data %>% 
                    left_join(external_state_encoded, by=c("DemW3", "wave")) %>%
                    rename(state=States) %>%
                    mutate(state = tolower(state)) %>%
                    left_join(external_county_selected, by=c("CVDID", "wave"), unmatched="drop", multiple='first') %>%
                    filter(sample == "core") %>%
                    filter(wave != "15b") %>%
                    left_join(external_county_extra, by=c("state","county"), multiple='first') %>%
                    select(all_of(vars)) %>%
                    group_by(state) %>% 
                    mutate(m_lat=mean(lat, na.rm=TRUE), m_lng=mean(lng, na.rm=TRUE), m_pop=mean(population, na.rm=TRUE), lat=replace(lat, which(is.na(lat)), first(m_lat)), lng=replace(lng, which(is.na(lng)), first(m_lng)), population=replace(population, which(is.na(population)), first(m_pop))) %>%
                    rename(Political_Views=DemM7, Age=prlfc_dem_age, Race=DemC9, Gender=DemC5, Education=DemC23, BDI=BDI_total_raw, PSS=PSS_Total, STAI=STAI_State_Tscore, Fear=Fear_COVID_raw, Emot_Support=NIH_TB_Emot_Support_uncorrTscore, Loneliness=NIH_TB_Loneliness_uncorrTscore, Neuroticism=NEO_N_z-score, Extraversion=NEO_E_z-score, Openness=NEO_O_z-score, Agreeableness=NEO_A_z-score, Conscientiousness=NEO_C_z-score) %>%
                    mutate(Race_AA=as.integer(Race==4), Race_A=as.integer(Race==2), Race_W=as.integer(Race==5), Gender=as.integer(Gender==1), Education=replace(Education, which(Education==10), NA))                 

psych_dem_data <- core_samples %>% select(c("CVDID", "wave", "Race_AA", "Race_A", "Race_W", "Age", "Income", "Income_v2", "Gender", "Education", "Political_Views", "BDI", "PSS", "STAI", "Fear", "Emot_Support", "Loneliness", "Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"))

covid_data <- core_samples %>% select(c("CVDID", "wave", "county", "state", "lat", "lng", "population", "cases_avg", "deaths_avg", "cases_avg_per_100k", "deaths_avg_per_100k", "slope_new_cases", "slope_new_deaths", "Mandatory_SAH"))

write.csv(core_samples, "output/v3_python/core_z.csv")

write.csv(psych_dem_data, "output/v3_python/raw_z.csv")
write.csv(covid_data, "output/v3_python/cvd_z.csv")


# test = core_samples %>% group_by(state) %>% mutate(m_lat=mean(lat, na.rm=TRUE), m_lng=mean(lng, na.rm=TRUE), lat=replace(lat, which(is.na(lat)), first(m_lat)), lng=replace(lng, which(is.na(lng)), first(m_lng)))

# test %>% filter(is.na(lat)) %>% select(c('CVDID','wave', 'state', 'county', 'lat', 'lng', 'population', 'cases_avg'))


# core_samples %>% filter(wave==1) %>% select(c('CVDID','wave', 'state', 'county', 'lat', 'lng', 'population', 'cases_avg'))

# core_samples %>% filter(is.na(lat)) %>% select(c('CVDID','wave', 'state', 'county', 'lat', 'lng', 'population', 'cases_avg'))

# test = core_samples %>% group_by(state) %>% summarize(m_lat=mean(lat, na.rm=TRUE), m_lng=mean(lng, na.rm=TRUE))
# test %>% filter(is.na(m_lat))

# test = external_county_extra %>% filter(state=='virginia') %>% select(c('county'))

# external_county %>% filter(state=='alaska') %>% select(c('county'))

# write.csv(core_samples, "data/selected_core_samples.csv")
                             

# core_samples %>% select(all_of(c("DemM7", "prlfc_dem_age", "DemC9", "DemC23","DemC5", "wave"))) %>%
#                     filter(wave=="2")

# covid_data %>% filter(is.na(cases_avg)) %>% select('state', 'county', 'lat', 'lng', 'cases_avg', 'deaths_avg')

# test1 = external_county_extra %>% filter(state_name=='Louisiana')
# test2 = external_county %>% filter(state=='louisiana')

# print(unique(test1$county))
# print(unique(test2$county))


# test2 %>% select(c("county", "cases_avg", "deaths_avg"))
