import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from constants import *

cleaned_data = pd.read_csv('../../output/raw.csv')

waves = np.unique(cleaned_data['wave'].values)

for w in waves:
    plt.clf()
    selected = cleaned_data.loc[cleaned_data['wave']==w, continuous+['Mandatory_SAH']]
    n = selected.shape[0]
    n_SAH = selected.loc[selected['Mandatory_SAH']==1,:].shape[0]
    sns_plt = sns.pairplot(selected, hue='Mandatory_SAH', markers=['+', 'x'], kind='reg', diag_kind='hist')
    sns_plt.fig.suptitle('wave='+str(w) + ' n='+str(n) + ' n_SAH='+str(n_SAH))
    sns_plt.savefig("../../output/figures/new_corners/"+str(w)+".png")