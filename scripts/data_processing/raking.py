import pickle
import pandas as pd
import numpy as np

from imputation_data_structure import *
from constants import *

def rake_data():

    with open('../../output/v3_python/core_imputed.pickle', 'rb') as f:
        core_imputed = pickle.load(f)

    with open('../../output/v3_python/cvd_imputed.pickle', 'rb') as f:
        cvd_imputed = pickle.load(f)

    print(core_imputed.meta)
    print(cvd_imputed.meta)
        
    path = '../../output/v3_python/core.csv'
    
    df = pd.read_csv(path, index_col=0)
    rake_weights = df.loc[df['wave']==1, 'rake_weights'].values

    m_rake_weights = np.tile(rake_weights[:, np.newaxis, np.newaxis], (1,14,16))

    remain_vars = core_imputed.imputed[:,:4,:]
    rake_vars = core_imputed.imputed[:,4:,:]

    raked = rake_vars * m_rake_weights

    raked_dat = np.concatenate([remain_vars, raked], axis=1)

    # Get the mandatory stay at home orders
    mandatory_SAH = np.empty([len(core_imputed.meta['CVDIDs']), 1, len(core_imputed.meta['waves'])])
    mandatory_SAH[:] = np.nan

    # Impute the subjects location
    state_matrix = np.empty([len(core_imputed.meta['CVDIDs']), len(core_imputed.meta['waves'])], dtype="<U20")
    state_matrix[:] = ''

    SAH_lookup = df.groupby(['wave', 'state']).first()['Mandatory_SAH']

    for i in range(len(core_imputed.meta['CVDIDs'])):
        for w in core_imputed.meta['waves']:
            state_val = df.loc[(df['CVDID']==core_imputed.meta['CVDIDs'][i]) & (df['wave']==w), 'state'].values
            SAH_val = df.loc[(df['CVDID']==core_imputed.meta['CVDIDs'][i]) & (df['wave']==w), 'Mandatory_SAH'].values
            if len(state_val) != 0:
                state_matrix[i, w-1] = state_val[0]
                mandatory_SAH[i, 0, w-1] = SAH_val[0]
            else:
                state_matrix[i, w-1] = state_matrix[i, w-2]
                mandatory_SAH[i, 0, w-1] = SAH_lookup.loc[(w-1, state_matrix[i, w-1])]

    full_dat = np.concatenate([raked_dat, cvd_imputed.imputed, mandatory_SAH], axis=1)
    column_names = core_imputed.meta['columns'] + cvd_imputed.meta['columns'] + ['Mandatory_SAH']
    cvdids = core_imputed.meta['CVDIDs']

    tmp_dict = {'data': full_dat, 'columns': column_names, 'cvdids': cvdids}

    print(np.isnan(full_dat).any())

    with open('../../output/v3_python/raked_full_imputed.pickle', 'wb') as handle:
        pickle.dump(tmp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    full_dat = np.concatenate([core_imputed.imputed, cvd_imputed.imputed, mandatory_SAH], axis=1)
    column_names = core_imputed.meta['columns'] + cvd_imputed.meta['columns'] + ['Mandatory_SAH']
    cvdids = core_imputed.meta['CVDIDs']

    tmp_dict = {'data': full_dat, 'columns': column_names, 'cvdids': cvdids}

    print(np.isnan(full_dat).any())

    with open('../../output/v3_python/unraked_full_imputed.pickle', 'wb') as handle:
        pickle.dump(tmp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == '__main__':
    rake_data()