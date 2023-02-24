import pickle
import pandas as pd
import numpy as np

from imputation_data_structure import *
from constants import *

with open('../../output/v3_python/core_imputed.pickle', 'rb') as f:
    core_imputed = pickle.load(f)

with open('../../output/v3_python/cvd_imputed.pickle', 'rb') as f:
    cvd_imputed = pickle.load(f)

def rake_data():
    path = '../../data/Wave1-16_paper_release.csv'

print(core_imputed.meta)