import numpy as np
import pandas as pd
import math
import recommenders as rec
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge

from constants import *


class Structure:
    def __init__(self, df, columns, dtype='covid-dynamic'):
        self.ori, self.meta = self.create_data_structure(df, columns, dtype)
        self.scaled_ori = self.ori.copy()
        self.imputed = self.ori.copy()
        self.is_scaled = False
        self.is_imputed = False
        self.impute_method_wave = None
        self.impute_method_subject = None
        self.scale_original()
    
    def create_data_structure(self, df, columns, dtype):
        """
        Create the full data structure for the analysis when initializing the data object
        """

        if dtype == 'covid-dynamic':
            waves = np.unique(df["wave"].values)
            
            CVDIDs = np.unique(df["CVDID"].values)
            
            # create an array to store the numerical values
            tmp_array = np.empty((df.loc[df['wave']==1,:].shape[0], len(columns), len(waves)))
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
        
        elif dtype == 'ucl-social-study':
            weeks = np.unique(df['week'].astype(int).values)

            record_ids = np.unique(df['record_id'].values)

            # create an array to store to numerical values
            tmp_array = np.empty([len(record_ids), len(columns), len(weeks)])
            tmp_array[:] = np.nan

            for i in range(len(record_ids)):
                for j in range(len(columns)):
                    for w in weeks:
                        tmp = df.loc[(df['week']==w) & (df['record_id']==record_ids[i]), columns[j]]
                        if len(tmp.values) != 0:
                            if (~np.isnan(tmp.values[0])) | (~math.isnan(tmp.values[0])):
                                tmp_array[i, j, w-1] = tmp.values[0]

            meta = {'waves': weeks,
                    'CVDIDs': record_ids,
                    'columns': columns}

        
            return tmp_array, meta
    
    def scale_original(self):
        
        for w in self.meta['waves']:
            wave_data = self.pull_by_wave(w)
            wave_data, indexes = self.generate_structure_for_imput(wave_data)
            m,n = wave_data.shape
            
            for i in range(n):
                wave_data[:, i] = self.scale_to_zero_one(wave_data[:, i], indexes[i], self.meta['columns'][indexes[i]])
            self.update_wave(wave_data, indexes, w)
        self.scaled_ori = self.imputed.copy()
        
    def generate_structure_for_imput(self, wave_data):
        """
        Generate the data for a specific wave for the imputation.
        """
        m,n = wave_data.shape
        missing_indexes = self.index_missing_var(wave_data)
            
        all_indexes = [x for x in range(n)]
        all_indexes = np.delete(all_indexes, missing_indexes)
        new_wave_data = np.take(wave_data, all_indexes, axis=1)
        return new_wave_data, all_indexes
    
    def impute_wave(self, new_wave_data, k, alpha=0.01, beta=0.01, iter=200, verbose=False):
        """
        Impute the missing data for the specific wave.
        """
        new_wave_data = np.nan_to_num(new_wave_data)
        imputer = rec.MatrixFactorization(new_wave_data, k, alpha=alpha, beta=beta, iterations=iter)
        training = imputer.train(verbose=verbose)
        fm = imputer.full_matrix()
        return fm, training
    
    def impute_subject(self, subject_data, k, alpha=0.01, beta=0.01, iter=200, verbose=False):
        """
        Impute the missing data for the specific subject.
        """
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
            self.imputed[:,indexes[i],wave-1] = fm[:, i]
            
    def update_subject(self, fm, subject_index):
        """
        Update the subject data after imputation
        """
        self.imputed[subject_index,:,:] = fm
    
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
        return self.imputed[:, :, wave-1].copy()
    
    def pull_by_subject(self, id_index):
        """
        Get the data from subject ID
        """
        return self.imputed[id_index, :, :].copy()
    
    def scale_to_zero_one(self, array, i, col_name):
        """
        A scaling function to scale an array between 0 and 1 using the max/min values
        """
        all_var_ranges = variable_ranges.keys()
        
        if col_name in all_var_ranges:
            max = variable_ranges[col_name][1]
            min = variable_ranges[col_name][0]
        else:
            max = np.nanmax(self.ori[:,i,:])
            min = np.nanmin(self.ori[:,i,:])

        return (array - min) / (max - min)
    
    def mf_imputation(self, k1, k2, alpha1, alpha2, beta1, beta2, iteration=2000, verbose="all", save_loss_wave=None, save_loss_subject=None):
        """
        run the imputation for the entire dataset
        """
        m, n, z = self.imputed.shape
        
        wave_training_loss = {}
        
        verbose_loss_wave = False
        verbose_loss_subject = False
        
        for i in range(1, z+1):
            if (verbose == "wave") | (verbose == "all") :
                print("=====imputation for wave "+ str(i) + " k={} =====".format(str(k1)))
                verbose_loss_wave = True
            wave_data, indexes = self.generate_structure_for_imput(self.pull_by_wave(i))
            imputed_wave_data, training_processes = self.impute_wave(wave_data, k=k1, alpha=alpha1, beta=beta1, iter=iteration, verbose=verbose_loss_wave)
            self.update_wave(imputed_wave_data, indexes, i)
            wave_training_loss[i] = training_processes
        
        if save_loss_wave:
            with open(save_loss_wave, 'wb') as handle:
                pickle.dump(wave_training_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                
        subject_training_loss = {}
            
        for i in range(m):
            if (verbose == "subject") | (verbose == "all"):
                print("=====imputation for subject "+ str(self.meta['CVDIDs'][i]) + " k={} =====".format(str(k2)))
                verbose_loss_subject = True
            imputed_subject_data, training_processes = self.impute_subject(self.pull_by_subject(i), k=k2, alpha=alpha2, beta=beta2, iter=iteration, verbose=verbose_loss_subject)
            self.update_subject(imputed_subject_data, i)
            subject_training_loss[i] = training_processes
            
        if save_loss_subject:
            with open(subject_training_loss, 'wb') as handle:
                pickle.dump(subject_training_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        self.is_imputed = True
        self.impute_method_wave = 'mf_k1={}'.format(str(k1))
        self.impute_method_subject = 'mf_k2={}'.format(str(k2))
        
    def mf_impute_wave(self, k, iter, reset=True, verbose=False):
        # Run imputation for mf for each wave
        if reset==True:
            self.reset_imputed()
            
        self.impute_method_wave = "mf_wave_k={}".format(str(k))
        
        for w in self.meta['waves']: 
            #print("=====imputation for wave "+ str(w) + " method={} =====".format(self.impute_method_wave))
            wave_data, indexes = self.generate_structure_for_imput(self.pull_by_wave(w))
            imputed_wave_data, _ = self.impute_wave(wave_data, k=k, iter=iter, verbose=verbose)
            self.update_wave(imputed_wave_data, indexes, w)
            
        self.is_imputed = True
        
    def knn_impute_wave(self, k, reset=True):
        if reset==True:
            self.reset_imputed()
            
        self.impute_method_wave = "knn_wave_k={}".format(str(k))
        
        imputer = KNNImputer(n_neighbors=k, weights='uniform')
            
        for w in self.meta['waves']: 
            #print("=====imputation for wave "+ str(w) + " method={} =====".format(self.impute_method_wave))
            wave_data, indexes = self.generate_structure_for_imput(self.pull_by_wave(w))
            imputed_wave_data = imputer.fit_transform(wave_data)
            self.update_wave(imputed_wave_data, indexes, w)
            
        self.is_imputed = True

    def simple_impute_wave(self, reset=False):
        if reset==True:
            self.reset_imputed()
            
        self.impute_method_wave = "simple_wave_method={}".format('mean')
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            
        for w in self.meta['waves']: 
            #print("=====imputation for wave "+ str(w) + " method={} =====".format(self.impute_method_wave))
            wave_data, indexes = self.generate_structure_for_imput(self.pull_by_wave(w))
            imputed_wave_data = imputer.fit_transform(wave_data)
            self.update_wave(imputed_wave_data, indexes, w)
            
        self.is_imputed = True

    def multiple_impute_wave(self, estimator='BayesianRidge', max_iter=100, reset=False):
        if reset==True:
            self.reset_imputed()
            
        self.impute_method_wave = "multiple_wave"
        if estimator=='Forest':
            est = RandomForestRegressor(
                # We tuned the hyperparameters of the RandomForestRegressor to get a good
                # enough predictive performance for a restricted execution time.
                n_estimators=50,
                n_jobs=2
            )
            imputer = IterativeImputer(estimator=est, max_iter=max_iter)
        elif estimator=='Ridge':
            imputer = IterativeImputer(estimator=Ridge(alpha=1e3),max_iter=max_iter)
        elif estimator=='BayesianRidge':
            imputer = IterativeImputer(max_iter=max_iter)

            
        for w in self.meta['waves']: 
            #print("=====imputation for wave "+ str(w) + " method={} =====".format(self.impute_method_wave))
            wave_data, indexes = self.generate_structure_for_imput(self.pull_by_wave(w))
            imputed_wave_data = imputer.fit_transform(wave_data)
            self.update_wave(imputed_wave_data, indexes, w)
            
        self.is_imputed = True

    def mf_impute_subject(self, k, iter, reset=True, verbose=False):
        if reset==True:
            self.reset_imputed()
            
        self.impute_method_subject = 'mf_subject_k={}'.format(str(k))
                    
        for i in range(len(self.meta['CVDIDs'])):
            tmp = self.pull_by_subject(i)
            waves = np.reshape(np.asarray(self.meta['waves']),(1,len(self.meta['waves'])))
            tmp = np.vstack([tmp, waves])
            detect_na_features = []
            for j in range(tmp.shape[0]):
                if np.isnan(tmp[j,:]).all():
                    detect_na_features.append(j)
            imputed_subject_data, training_processes = self.impute_subject(tmp, k=k, iter=iter, verbose=verbose)
            for idx in detect_na_features:
                imputed_subject_data[idx, :] = np.nan

            imputed_subject_data = imputed_subject_data[:-1,:]
            self.update_subject(imputed_subject_data, i)
        
        self.is_imputed = True
        
    def knn_impute_subject(self, k, reset=True):
        if reset==True:
            self.reset_imputed()
            
        self.impute_method_subject = "knn_subject_k={}".format(str(k))
        
        imputer = KNNImputer(n_neighbors=k, weights='uniform', keep_empty_features=True)
        
        for i in range(len(self.meta['CVDIDs'])):
            tmp = self.pull_by_subject(i)
            waves = np.reshape(np.asarray(self.meta['waves']),(1,len(self.meta['waves'])))
            tmp = np.vstack([tmp, waves])
            detect_na_features = []
            for j in range(tmp.shape[0]):
                if np.isnan(tmp[j,:]).all():
                    detect_na_features.append(j)

            imputed_subject_data = imputer.fit_transform(tmp.T).T

            for idx in detect_na_features:
                imputed_subject_data[idx, :] = np.nan
            imputed_subject_data = imputed_subject_data[:-1,:]
            
            self.update_subject(imputed_subject_data, i)
            
        self.is_imputed = True

    def simple_impute_subject(self, reset=True):
        if reset==True:
            self.reset_imputed()
            
        self.impute_method_subject = "simple_subject_method=mean"
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean', keep_empty_features=True)
        
        for i in range(len(self.meta['CVDIDs'])):
            tmp = self.pull_by_subject(i)
            waves = np.reshape(np.asarray(self.meta['waves']),(1,len(self.meta['waves'])))
            tmp = np.vstack([tmp, waves])
            detect_na_features = []
            for j in range(tmp.shape[0]):
                if np.isnan(tmp[j,:]).all():
                    detect_na_features.append(j)
                    

            imputed_subject_data = imputer.fit_transform(tmp.T).T
            for idx in detect_na_features:
                imputed_subject_data[idx, :] = np.nan
            
            imputed_subject_data = imputed_subject_data[:-1,:]
            
            self.update_subject(imputed_subject_data, i)
            
        self.is_imputed = True

    def multiple_impute_subject(self, max_iter=100, reset=True):
        if reset==True:
            self.reset_imputed()
            
        self.impute_method_subject = "multiple_subject={}"
        
        imputer = IterativeImputer(max_iter=max_iter, keep_empty_features=True)
        
        for i in range(len(self.meta['CVDIDs'])):
            tmp = self.pull_by_subject(i)
            waves = np.reshape(np.asarray(self.meta['waves']),(1,len(self.meta['waves'])))
            tmp = np.vstack([tmp, waves])
            detect_na_features = []
            for j in range(tmp.shape[0]):
                if np.isnan(tmp[j,:]).all():
                    detect_na_features.append(j)
                    

            imputed_subject_data = imputer.fit_transform(tmp.T).T
            for idx in detect_na_features:
                imputed_subject_data[idx, :] = np.nan
            
            imputed_subject_data = imputed_subject_data[:-1,:]
            
            self.update_subject(imputed_subject_data, i)
            
        self.is_imputed = True
        
    def reset_imputed(self):
        """Reset the imputed dataframe
        """
        self.imputed = self.ori.copy()
        self.is_imputed = False
        self.scale_original()
    
    def generate_random_samples(self, n_samples):
        """Generate a random sample for testing the bias of a specific imputation method

        Args:
            n_samples (int): the number of samples that we need.

        Returns:
            test_case_indexes: A list of 3d tuples for the indexes for the sample sample.
        """
        ori_NAs_indicator = np.isnan(self.ori)
        # Create 100 testing cases for the imputation for each wave
        m,n,l = self.imputed.shape
        test_case_indexes = []
        
        i=0
        while (i < n_samples):
            #print(i)
            this_index = (np.random.randint(0,m) , np.random.randint(0,n), np.random.randint(0,l))
            if (ori_NAs_indicator[this_index] == True) | (this_index in test_case_indexes):
                continue
            test_case_indexes.append(this_index)
            i += 1
         
        return test_case_indexes
    
    def create_testing_data(self, n_samples=100):
        """ Create a dataset for for the testing data

        Args:
            n_samples (int, optional): the number of samples. Defaults to 100.

        Returns:
            test_case_indexes (list): A list of 3d tuples for the indexes for the sample sample.
            test_case_values (list): A list for the values of the testing data
        """
        # Create 100 testing cases for the imputation for each wave[]
        test_case_values = []
        test_case_indexes = self.generate_random_samples(n_samples=n_samples)
        for idx in test_case_indexes:
            this_value = self.imputed[idx]
            test_case_values.append(this_value)
        
        return test_case_indexes, test_case_values
    
    def set_na_values(self, test_case_indexes):
        """Set target values to NA

        Args:
            test_case_indexes (list): A list of indexes that are set to NA 
        """
        for idx in test_case_indexes:
            self.imputed[idx] = np.nan
    
    def estimate_imputation_error(self, test_case_indexes):
        """A function to estimate the bias of a paritular imputation method

        Args:
            test_case_indexes (list): List of indexes for the testing data

        Raises:
            ValueError: Value error for non-imputed data

        Returns:
            mse: mean squared error for the dataset
        """
        mse = 0
        test_case_count = 0
        if self.is_imputed == True:
            for idx in test_case_indexes:
                if ~(np.isnan(self.imputed[idx])):
                    test_case_count += 1
                    mse += (self.scaled_ori[idx] - self.imputed[idx])**2

        elif self.is_imputed == False:
            raise ValueError('Data have not imputed yet.')
        
        return mse/test_case_count, test_case_count
    
    def knn_imputation(self, k1=5, k2=1, verbose="wave"):
        """KNN imputation

        Args:
            k1 (int, optional): K to impute wave. Defaults to 5.
            k2 (int, optional): K to impute subject. Defaults to 1. Set to default 1 because we assumes that the missing imputation should be similar to the closest wave.
            verbose (str, optional): Show the details. Defaults to "wave". "subject" or "all".
        """
        m, n, z = self.imputed.shape
        
        imputer_k1 = KNNImputer(n_neighbors=k1, weights='uniform')
        for i in range(1, z+1):
            if (verbose == "wave") | (verbose == "all") :
                print("=====imputation for wave "+ str(i) + " k={} =====".format(str(k1)))
            wave_data, indexes = self.generate_structure_for_imput(self.pull_by_wave(i))
            
            imputed_wave_data = imputer_k1.fit_transform(wave_data)
            self.update_wave(imputed_wave_data, indexes, i)
            
        
        imputer_k2 = KNNImputer(n_neighbors=k2, weights='uniform')
        for i in range(m):
            if (verbose == "subject") | (verbose == "all"):
                print("=====imputation for subject "+ str(self.meta['CVDIDs'][i]) + " k={} =====".format(str(k2)))
                
            imputed_subject_data = imputer_k2.fit_transform(self.pull_by_subject(i).T).T
            self.update_subject(imputed_subject_data, i)
        
        self.is_imputed = True
        self.impute_method_wave = 'knn_k1={}'.format(str(k1))
        self.impute_method_subject = 'knn_k2={}'.format(str(k2))
            
    def get_targeted_indexes(self, test_indexes, selected_vars):
        indexes = []
        for var in selected_vars:
            indexes.append(self.meta['columns'].index(var))

        selected_indexes = []
        for idx in test_indexes:
            if idx[1] in indexes:
                selected_indexes.append(idx)
        return selected_indexes

        
        
            
        
                
        
        
        
        
        
        