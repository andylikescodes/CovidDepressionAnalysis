import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl
from data_manipulation import *
from constants import *
from tqdm import tqdm

def create_corner_plots(pkl_data, save_path):
    
    for i in tqdm(range(1, 17)):
        print("======Plotting wave"+str(i))

        plt.clf()
        wave = create_wave_df(pkl_data, i, variable_selected)
        n = wave.shape[0]
        n_SAH = wave.loc[wave['Mandatory_SAH']==1,:].shape[0]
        sns_plt = sns.pairplot(wave, hue='Mandatory_SAH', markers=['+', 'x'], kind='reg', diag_kind='hist')
        sns_plt.fig.subplots_adjust(top=0.95)
        sns_plt.fig.suptitle('wave='+str(i) + ' n='+str(n) + ' n_SAH='+str(n_SAH), fontsize=25)
        path = "../../output/figures/{}/".format(save_path)+str(i)+".png"
        print('save to '+ path)
        sns_plt.savefig(path)


if __name__ == '__main__':    

    with open('../../output/v3_python/full_imputed.pickle', 'rb') as f:
        full_imputed = pkl.load(f)

    create_corner_plots(full_imputed, "Corrected_full_imputed_corners")

    