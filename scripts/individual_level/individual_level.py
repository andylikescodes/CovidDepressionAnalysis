import numpy as np
import pandas as pd
import pickle as pkl

def detect_change_in_SAH():
    with open('../../output/v3_python/full_imputed.pickle', 'rb') as f:
        full_imputed = pkl.load(f)

    print(full_imputed['columns'])

    SAH_index = full_imputed['columns'].index('Mandatory_SAH')
    BDI_index = full_imputed['columns'].index('BDI')

    SAH_data = full_imputed['data'][:,SAH_index,:]
    depression_data = full_imputed['data'][:,BDI_index,:]

    under_SAH = []
    no_SAH = []

    for i in range(SAH_data.shape[0]):
        subject_SAH = SAH_data[i,:]
        for j in range(len(subject_SAH)-1):
            if subject_SAH[j] != subject_SAH[j+1]:
                if subject_SAH[j] == 1:
                    under_SAH.append(depression_data[i, j])
                    no_SAH.append(depression_data[i,j+1])
                elif subject_SAH[j+1] == 1:
                    under_SAH.append(depression_data[i, j+1])
                    no_SAH.append(depression_data[i,j])

    return (under_SAH, no_SAH)



under_SAH, no_SAH = detect_change_in_SAH()

flag_SAH = [True for i in range(len(under_SAH))]
flag_no_SAH = [False for i in range(len(no_SAH))]

BDI_vals = np.concatenate([under_SAH, no_SAH])
flags = np.concatenate([flag_SAH, flag_no_SAH])

df = pd.DataFrame({'Mandatory_SAH': flags, 'BDI': BDI_vals})
df.to_csv('../../output/v3_python/BDI_longitudinal.csv')

print(df)

print(len(under_SAH))
print(len(no_SAH))

print(np.mean(under_SAH))
print(np.std(under_SAH)/len(under_SAH))

print(np.mean(no_SAH))
print(np.std(no_SAH)/len(no_SAH))