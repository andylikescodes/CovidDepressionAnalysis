import numpy as np
import pandas as pd
import math
import recommenders as rec

from constants import *


class Structure:
    def __init__(self, df):
        self.data, self.meta = self.create_data_structure(df)
    
    def create_data_structure(self, df):
        """
        Create the full data structure for the analysis when initializing the data object
        """
        waves = np.unique(df["wave"].values)
        
        CVDIDs = np.unique(df["CVDID"].values)
        
        columns = binaries + continuous
        
        # create an array to store the numerical values
        tmp_array = np.empty((df.loc[df['wave']==1,:].shape[0], len(columns), 16))
        tmp_array[:] = np.nan
        
        for i in range(len(CVDIDs)):
            for j in range(len(columns)):
                for w in waves:
                    tmp = df.loc[(df["wave"]==w) & (df["CVDID"]==CVDIDs[i]), columns[j]]
                    if len(tmp.values) != 0:
                        if (~np.isnan(tmp.values[0])) | (~math.isnan(tmp.values[0])):
                            tmp_array[i, j, w-1] = tmp.values[0]
                            
        meta = {'waves': waves,
                'CVDIDs': CVDIDs,
                'columns': columns}
        
        return tmp_array, meta
    
    def generate_structure_for_imput(self, wave_data):
        """
        Generate the data for a specific wave for the imputation.
        """
        m,n = wave_data.shape
        missing_indexes = self.index_missing_var(wave_data)
        print(missing_indexes)
        all_indexes = [x for x in range(n)]
        all_indexes = np.delete(all_indexes, missing_indexes)
        print(all_indexes)
        new_wave_data = np.take(wave_data, all_indexes, axis=1)
        return new_wave_data, all_indexes
    
    def impute_wave(self, new_wave_data, k, alpha=0.01, beta=0.01, iter=200):
        """
        Impute the missing data for the specific wave.
        """
        for i in range(new_wave_data.shape[1]):
            new_wave_data[:, i] = self.scale_to_zero_one(new_wave_data[:, i])
            
        new_wave_data = np.nan_to_num(new_wave_data)
        imputer = rec.MatrixFactorization(new_wave_data, k, alpha=alpha, beta=beta, iterations=iter)
        training = imputer.train()
        fm = imputer.full_matrix()
        return fm
        
    
    def index_missing_var(self, wave_data):
        """
        Get get columns with all missing values
        """
        indexes = []
        m, n = wave_data.shape
        for i in range(n):
            if (np.isnan(wave_data[:, i]).all()):
                indexes.append(i)
        return indexes
    
    def pull_by_wave(self, wave):
        """
        Get the wave data from the full data structure.
        """
        return self.data[:, :, wave-1]
    
    def scale_to_zero_one(self, array):
        """
        A scaling function to scale an array between 0 and 1 using the max/min values
        """
        max = np.nanmax(array)
        min = np.nanmin(array)
        
        return (array - min) / (max - min)
    
    
