binaries = ["Race_A", 
            "Race_AA", 
            "Race_W", 
            "Gender", 
            "Mandatory_SAH"]
continuous = ["Political_Views", 
              "Age", 
              "Education", 
              "BDI", 
              "PSS", 
              "State_Anxiety", 
              "Fear_COVID", 
              'Emo_Support',
              "Loneliness", 
              "Covid_Cases_State_Slope", 
              "Covid_Deaths_State_Slope",
              "Covid_Cases_County_avg",
              "Covid_Deaths_County_avg",
              "Covid_Cases_County_avg_per_100k",
              "Covid_Deaths_County_avg_per_100k"]
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