import numpy as np
import pandas as pd
from constants import *

path = "../../data/selected_core_samples.csv"
data = pd.read_csv(path,index_col=0)

wave_1 = data.loc[data["wave"]==1, :]

data.loc[:, 'Race_AA'] = (data.loc[:, 'DemC9'] == 4).astype('int')
data.loc[:, 'Race_A'] = (data.loc[:, 'DemC9'] == 2).astype('int')
data.loc[:, 'Race_W'] = (data.loc[:, 'DemC9'] == 5).astype('int')
data.loc[:, 'Gender'] = (data.loc[:, 'DemC5'] == 1).astype('int')

cleaned_data = data.drop(["DemC9", "DemC5"], axis=1).rename(name_mapping, axis=1)

# save data
cleaned_data.to_csv('../../output/v3_python/raw.csv', index=False)