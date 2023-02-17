import numpy as np
import pandas as pd
import math
import recommenders as rec
import pickle

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
        print('The columns that are missing for this wave: ')
        print(missing_indexes)
        for i in missing_indexes:
            print(self.meta['columns'][i])
            
        all_indexes = [x for x in range(n)]
        all_indexes = np.delete(all_indexes, missing_indexes)
        print(all_indexes)
        new_wave_data = np.take(wave_data, all_indexes, axis=1)
        return new_wave_data, all_indexes
    
    def impute_wave(self, new_wave_data, k, alpha=0.01, beta=0.01, iter=200, verbose=False):
        """
        Impute the missing data for the specific wave.
        """
        for i in range(new_wave_data.shape[1]):
            new_wave_data[:, i] = self.scale_to_zero_one(new_wave_data[:, i])
            
        new_wave_data = np.nan_to_num(new_wave_data)
        imputer = rec.MatrixFactorization(new_wave_data, k, alpha=alpha, beta=beta, iterations=iter)
        training = imputer.train(verbose=verbose)
        fm = imputer.full_matrix()
        return fm, training
    
    def impute_subject(self, subject_data, k, alpha=0.01, beta=0.01, iter=200, verbose=False):
        """
        Impute the missing data for the specific subject.
        """
        for i in range(subject_data.shape[1]):
            subject_data[:, i] = self.scale_to_zero_one(subject_data[:, i])
        
        subject_data = np.nan_to_num(subject_data)
        imputer = rec.MatrixFactorization(subject_data, k, alpha=alpha, beta=beta, iterations=iter)
        training = imputer.train(verbose=verbose)
        fm = imputer.full_matrix()
        return fm, training
    
    def update_wave(self, fm, indexes, wave):
        """
        Update the wave data after imputation
        """
        for i in range(fm.shape[1]):
            self.data[:,indexes[i],wave-1] = fm[:, i]
            
    def update_subject(self, fm, subject_index):
        """
        Update the subject data after imputation
        """
        self.data[subject_index,:,:] = fm
    
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
        return self.data[:, :, wave-1].copy()
    
    def pull_by_subject(self, id_index):
        """
        Get the data from subject ID
        """
        return self.data[id_index, :, :].copy()
    
    def scale_to_zero_one(self, array):
        """
        A scaling function to scale an array between 0 and 1 using the max/min values
        """
        max = np.nanmax(array)
        min = np.nanmin(array)
        
        return (array - min) / (max - min)
    
    def run_imputation(self, k1, k2, alpha1, alpha2, beta1, beta2, iteration=2000, verbose="wave", save_loss_wave=None, save_loss_subject=None):
        """
        run the imputation for the entire dataset
        """
        m, n, z = self.data.shape
        
        wave_training_loss = {}
        
        for i in range(1, z+1):
            if (verbose == "wave") | (verbose == "all") :
                print("=====imputation for wave "+ str(i) + "=====")
            wave_data, indexes = self.generate_structure_for_imput(self.pull_by_wave(i))
            imputed_wave_data, training_processes = self.impute_wave(wave_data, k=k1, alpha=alpha1, beta=beta1, iter=iteration, verbose=False)
            self.update_wave(imputed_wave_data, indexes, i)
            wave_training_loss[i] = training_processes
        
        if save_loss_wave:
            with open(save_loss_wave, 'wb') as handle:
                pickle.dump(wave_training_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                
        subject_training_loss = {}
            
        for i in range(m):
            if (verbose == "subject") | (verbose == "all"):
                if ((i+1) % 100 == 0) | ((i+1)==len(m)):
                    print("=====imputation for subject "+ str(self.meta['CVDIDs'][i]) + "=====")
            imputed_subject_data, training_processes = self.impute_subject(self.pull_by_subject(i), k=k2, alpha=alpha2, beta=beta2, iter=iteration, verbose=False)
            self.update_subject(imputed_subject_data, i)
            subject_training_loss[i] = training_processes
            
        if save_loss_subject:
            with open(save_loss_wave, 'wb') as handle:
                pickle.dump(subject_training_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)