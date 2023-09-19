# UCL social studies
UCL_personality = {
    'Neuroticism': ['pers1',
                    'pers2',
                    'pers3'],
    'Extraversion': ['pers4',
                    'pers5',
                    'pers6'],
    'Openness': ['pers7',
                'pers8',
                'pers9'],
    'Agreeableness': ['pers10',
                    'pers11',
                    'pers12'],
    'Conscientiousness': ['pers13',
                        'pers14',
                        'pers15']
}

UCL_depression = [
    'phq1',
    'phq2',
    'phq3',
    'phq4',
    'phq5',
    'phq6',
    'phq7',
    'phq8',
    'phq9',
    'phqextra'
]

UCL_anxiety = [
    'gad1',
    'gad2',
    'gad3',
    'gad4',
    'gad5',
    'gad6',
    'gad7'
]

UCL_covid_anxiety = [
    'cas1',
    'cas2',
    'cas3',
    'cas4',
    'cas5'
]

UCL_stressors = [
    'stressorsminor1',
    'stressorsminor2',
    'stressorsminor3',
    'stressorsminor4',
    'stressorsminor5',
    'stressorsminor6',
    'stressorsminor7',
    'stressorsminor8',
    'stressorsminor9',
    'stressorsminor10',
    'stressorsminor11',
    'stressorsminor12',
    'stressorsminor13',
    'stressorsminor14',
    'stressorsminor15',
    'stressorsminor16',
    'stressorsmajor1',
    'stressorsmajor2',
    'stressorsmajor3',
    'stressorsmajor4',
    'stressorsmajor5',
    'stressorsmajor6',
    'stressorsmajor7',
    'stressorsmajor8',
    'stressorsmajor9',
    'stressorsmajor10',
    'stressorsmajor11',
    'stressorsmajor12',
    'stressorsmajor13',
    'stressorsmajor14',
    'stressorsmajor15',
    'stressorsmajor16'
]

UCL_loneliness = [
    'soc1',
    'soc2',
    'soc3',
    'soc4'
]

UCL_support = [
    'supp1',
    'supp2',
    'supp3',
    'supp4',
    'supp5',
    'supp6'
]

UCL_demographic = [
    'age',
    'gender',
    'ethnic',
    'marital',
    'education',
    'income'
]


## End of UCL social study thing

# Covid study thing
variable_ranges = {
    'Political_Views': [1, 7],
    'Age': [18, 82],
    'Education': [1, 9],
    'Income': [1, 6],
    'Income_v2': [1, 8],
    'BDI': [0, 63],
    'PSS': [0, 40],
    'STAI': [20, 80],
    'Fear': [0, 28],
    'Emot_Support': [8, 40],
    'Loneliness': [5, 25],
    "NEO_Neuroticism": [0, 48], 
    "NEO_Extraversion": [0, 48], 
    "NEO_Openness": [0, 48], 
    "NEO_Agreeableness": [0, 48], 
    "NEO_Conscientiousness": [0, 48]
}

# wave weeks
wave_weeks = {1: 1,
              2: 2,
              3: 3,
              4: 4,
              5: 6,
              6: 8,
              7: 10,
              8: 13,
              9: 16,
              10: 19,
              11: 22,
              12: 25,
              13: 28,
              14: 32,
              15: 36,
              16: 43}

variable_selected = ['Political_Views',
                    'Age',
                    'Education',
                    'Income',
                    'Income_v2',
                    'BDI',
                    'PSS',
                    'STAI',
                    'Fear',
                    'Emot_Support',
                    'Loneliness',
                    "NEO_Neuroticism", 
                    "NEO_Extraversion", 
                    "NEO_Openness", 
                    "NEO_Agreeableness", 
                    "NEO_Conscientiousness",
                    'cases_avg',
                    'deaths_avg',
                    'cases_avg_per_100k',
                    'deaths_avg_per_100k',
                    'slope_new_cases',
                    'slope_new_deaths',
                    'population',
                    'Mandatory_SAH']


binaries = ["Race_A", 
            "Race_AA", 
            "Race_W", 
            "Gender"]

continuous = ["Political_Views", 
              "Age", 
              "Education",
              "Income",
              "BDI", 
              "PSS", 
              "State_Anxiety", 
              "Fear_COVID", 
              'Emot_Support',
              "Loneliness", 
              "Covid_Cases_State_Slope", 
              "Covid_Deaths_State_Slope",
              "Covid_Cases_County_avg",
              "Covid_Deaths_County_avg",
              "Covid_Cases_County_avg_per_100k",
              "Covid_Deaths_County_avg_per_100k"]

with_binaries = ["Race_A", 
                "Race_AA", 
                "Race_W", 
                "Gender",
                "Political_Views", 
                "Age", 
                "Education",
                "Income", 
                "BDI", 
                "PSS", 
                "STAI", 
                "Fear", 
                'Emot_Support',
                "Loneliness",
                "NEO_Neuroticism", 
                "NEO_Extraversion", 
                "NEO_Openness", 
                "NEO_Agreeableness", 
                "NEO_Conscientiousness"]

with_binaries_no_race = ["Race_A", 
                "Race_AA", 
                "Race_W", 
                "Gender",
                "Political_Views", 
                "Age", 
                "Education", 
                "BDI", 
                "PSS", 
                "STAI", 
                "Fear", 
                'Emot_Support',
                "Loneliness",
                "NEO_Neuroticism", 
                "NEO_Extraversion", 
                "NEO_Openness", 
                "NEO_Agreeableness", 
                "NEO_Conscientiousness"]

without_binaries = [
                "Political_Views", 
                "Age", 
                "Education",
                "Income", 
                "Income_v2",
                "BDI", 
                "PSS", 
                "STAI", 
                "Fear", 
                'Emot_Support',
                "Loneliness",
                "NEO_Neuroticism", 
                "NEO_Extraversion", 
                "NEO_Openness", 
                "NEO_Agreeableness", 
                "NEO_Conscientiousness"]

continuous_covid = ["cases_avg", 
                    "deaths_avg", 
                    "cases_avg_per_100k", 
                    "deaths_avg_per_100k",
                    "slope_new_cases", 
                    "slope_new_deaths", 
                    "lat", 
                    'lng', 
                    "population"]


all = binaries + continuous

others = ["CVDID", 
          "DemW3",
          "States",
          "wave"]


name_mapping = {'DemM7': 'Political_Views',
                'DemC23': 'Education',
                'prlfc_dem_age': "Age",
                'BDI_total_raw': "BDI",
                'PSS_Total': "PSS",
                'STAI_State_raw': "State_Anxiety",
                'Fear_COVID_raw': "Fear_COVID",
                'NIH_TB_Emot_Support_raw_total': "Emo_Support",
                'NIH_TB_Loneliness_raw_total': "Loneliness",
                "slope_new_cases": "Covid_Cases_State_Slope",
                "slope_new_deaths": "Covid_Deaths_State_Slope",
                "cvd_cases_7d_avg": "Covid_Cases_County_avg",
                "cvd_deaths_7d_avg": "Covid_Deaths_County_avg",
                "cvd_cases_7d_avg_per_100k": "Covid_Cases_County_avg_per_100k",
                "cvd_deaths_7d_avg_per_100k": "Covid_Deaths_County_avg_per_100k"}