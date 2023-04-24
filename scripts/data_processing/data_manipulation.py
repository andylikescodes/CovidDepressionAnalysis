import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from constants import *

def find_index_of_selected_columns(all_var, var_selected):
    indexes = []
    for i in range(len(all_var)):
        if all_var[i] in var_selected:
            indexes.append(i)
    return indexes


def create_wave_df(full_data, w, var_selected):
    indexes = find_index_of_selected_columns(full_data['columns'], var_selected)

    data = full_data['data'][:, indexes, w-1]

    df = pd.DataFrame(data, columns=var_selected)
    return df

def combine_data_sources():
    with open('../../output/v3_python/core_imputed.pickle', 'rb') as f:
        core_imputed = pkl.load(f)

    with open('../../output/v3_python/cvd_imputed.pickle', 'rb') as f:
        cvd_imputed = pkl.load(f)

    df_raw2 = pd.read_csv('../../output/v3_python/raw2.csv')
    df = pd.read_csv('../../output/v3_python/core.csv')

    matrix_for_binary = np.empty((len(core_imputed.meta['CVDIDs']),len(binaries),len(core_imputed.meta['waves'])))
    matrix_for_binary[:] = np.nan

    for i in range(len(core_imputed.meta['CVDIDs'])):
        for j in range(len(binaries)):
            for w in range(len(core_imputed.meta['waves'])):
                val = df_raw2.loc[(df_raw2['wave']==w+1) & (df_raw2['CVDID']==core_imputed.meta['CVDIDs'][i]), binaries[j]].values
                if len(val) == 0:
                    matrix_for_binary[i,j,w] = matrix_for_binary[i,j,w-1]
                else:
                    matrix_for_binary[i,j,w] = val[0]

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

    full_dat = np.concatenate([matrix_for_binary, core_imputed.imputed, cvd_imputed.imputed, mandatory_SAH], axis=1)
    column_names = binaries + core_imputed.meta['columns'] + cvd_imputed.meta['columns'] + ['Mandatory_SAH']
    cvdids = core_imputed.meta['CVDIDs']

    tmp_dict = {'data': full_dat, 'columns': column_names, 'cvdids': cvdids}

    print(np.isnan(full_dat).any())

    with open('../../output/v3_python/full_imputed.pickle', 'wb') as handle:
        pkl.dump(tmp_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

def create_data_for_jasp():
    with open('../../output/v3_python/full_imputed.pickle', 'rb') as f:
        full_imputed = pkl.load(f)

    for i in tqdm(range(1, 17)):
        print("======creating wave"+str(i))
        wave = create_wave_df(full_imputed, i, full_imputed['columns'])
        wave['Mandatory_SAH'] = wave['Mandatory_SAH'] == 1
        wave.to_csv('../../output/jasp/wave'+str(i)+'.csv')

def combine_all_data_for_jasp():
    with open('../../output/v3_python/full_imputed.pickle', 'rb') as f:
        full_imputed = pkl.load(f)

    all_waves = []

    for i in tqdm(range(2, 13)):
        print("======creating wave"+str(i))
        wave = create_wave_df(full_imputed, i, full_imputed['columns'])
        wave['wave'] = i
        wave['ids'] = full_imputed['cvdids'].astype(str)
        all_waves.append(wave)
    
    df_all_waves = pd.concat(all_waves, axis=0)

    df_all_waves.to_csv('../../output/jasp/all.csv')

    
if __name__ == '__main__':
    combine_data_sources()
    create_data_for_jasp()
    combine_all_data_for_jasp()