from imputation_data_structure import *
import pandas as pd
import math
from random import sample, choices
from constants import *
import sys

class TestClass:
    def test_object(self):
        
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = test_df.loc[(test_df['wave']==1) | (test_df['wave']==2), :]
        test_structure = Structure(self.test_df)
        
        # generate 100 random test for the content of the data
        n_waves = [x for x in range(len(test_structure.meta['waves']))]
        n_columns = [x for x in range(len(test_structure.meta['columns']))]
        n_ids = [x for x in range(len(test_structure.meta['CVDIDs']))]
        
        test_cases = 100
        
        test_waves = choices(n_waves, k=test_cases)
        test_columns = choices(n_columns, k=test_cases)
        test_ids = choices(n_ids, k=test_cases)

        
        for i in range(len(test_waves)):
            out_val = test_structure.data[test_ids[i], test_columns[i], test_waves[i]]
            print(out_val)
            ori_val = self.test_df.loc[(test_df['wave']==test_structure.meta['waves'][test_waves[i]]) & (test_df['CVDID']==test_structure.meta['CVDIDs'][test_ids[i]]), test_structure.meta['columns'][test_columns[i]]].values
            print(ori_val)
            if len(ori_val) == 0:
                ori_val = np.float64(np.nan)
                print(1)
            elif math.isnan(ori_val[0]):
                ori_val = np.float64(np.nan)
                print(2)
            else:
                ori_val = ori_val[0]
                print(3)
            
            if (np.isnan(out_val)):
                assert np.isnan(out_val) == np.isnan(ori_val)
            else:
                assert out_val == ori_val
    
    def test_pull_wave(self):
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = test_df.loc[(test_df['wave']==1) | (test_df['wave']==2), :]
        test_structure = Structure(test_df)
        wave_data = test_structure.pull_by_wave(1)
        indexes = test_structure.index_missing_var(wave_data)
        assert test_structure.meta['columns'][indexes[0]] == 'BDI'
        
    def test_generate_structure_to_imput(self):
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = test_df.loc[(test_df['wave']==1) | (test_df['wave']==2), :]
        test_structure = Structure(test_df)
        wave_data = test_structure.pull_by_wave(1)
        new_wave_data, indexes = test_structure.generate_structure_for_imput(wave_data)
        assert new_wave_data.shape[1] == len(indexes)
        
        test_indexes = test_structure.index_missing_var(new_wave_data)
        assert len(test_indexes) == 0
        
    def test_impute_wave(self):
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = test_df.loc[(test_df['wave']==1) | (test_df['wave']==2), :]
        test_structure = Structure(test_df)
        wave_data = test_structure.pull_by_wave(1)
        new_wave_data, indexes = test_structure.generate_structure_for_imput(wave_data)
        
        imputed_wave_data = test_structure.impute_wave(new_wave_data, k=5, iter=1000)
        assert imputed_wave_data.shape == new_wave_data.shape
        assert np.isnan(imputed_wave_data).any() == False
        
    def test_impute_subject(self):
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = test_df.loc[(test_df['CVDID']==2) | (test_df['CVDID']==3), :]
        test_structure = Structure(test_df)
        subject_data = test_structure.pull_by_subject(1)
                
        imputed_subject_data = test_structure.impute_subject(subject_data, k=5, iter=1000)
        assert imputed_subject_data.shape == subject_data.shape
        assert imputed_subject_data.shape[1] == 16
        assert np.isnan(imputed_subject_data).any() == False

    def test_impute_full(self):
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_structure = Structure(test_df)
        test_structure.run_imputation(k1=15, k2=10, alpha1=0.01, alpha2=0.01, beta1=0.01, beta2=0.01, iteration=200, save_loss_wave='../../output/training_loss/wave.pkl', save_loss_subject='../../output/training_loss/subject.pkl')
        assert np.isnan(test_structure.data).any() == False
        
    def test_update_wave(self):
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = test_df.loc[(test_df['wave']==1) | (test_df['wave']==2), :]
        test_structure = Structure(test_df)
        
        wave_data = test_structure.pull_by_wave(2)
        print(test_structure.data.shape)
        print(wave_data.shape)
        
        new_wave_data, indexes = test_structure.generate_structure_for_imput(wave_data)
        
        imputed_wave_data = test_structure.impute_wave(new_wave_data, k=5, iter=1000)
        
        print(imputed_wave_data.shape)
        
    def test_create_test_indexes(self):
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = test_df.loc[(test_df['wave']==1) | (test_df['wave']==2), :]
        test_structure = Structure(test_df)
        samples = test_structure.generate_random_samples(n_samples=100)
        print(samples)
        print(len(samples))
        
    def test_testing_values(self):
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = test_df.loc[(test_df['wave']==1) | (test_df['wave']==2), :]
        test_structure = Structure(test_df)
        test_case_indexes, test_case_values = test_structure.create_testing_data(n_samples=100)
        
        print(test_case_indexes)
        print(len(test_case_values))
        print(test_case_values[0])
        
        for i in range(len(test_case_indexes)):
            assert (test_structure.ori[test_case_indexes[i]] == test_case_values[i])
        
    def test_estimate_imputation_error(self):
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = test_df.loc[(test_df['wave']==1) | (test_df['wave']==2), :]
        test_structure = Structure(test_df)
        test_case_indexes, test_case_values = test_structure.create_testing_data(n_samples=100)
        
        test_structure.run_imputation(k1=10, k2=5, alpha1=0.01, alpha2=0.01, beta1=0.01, beta2=0.01, iteration=3000, verbose='wave')
        assert(test_structure.is_imputed == True)
        print(test_structure.estimate_imputation_error(test_case_indexes))
        
        
    def test_compare_different_computation(self):
        test_df = pd.read_csv('../../output/v3_python/raw.csv')
        # Focus on just two waves for testing
        test_df = test_df.loc[(test_df['wave']==1) | (test_df['wave']==2), :]
        test_structure = Structure(test_df)
        test_case_indexes, test_case_values = test_structure.create_testing_data(n_samples=100)
        
        test_structure.reset_imputed()
        test_structure.set_na_values(test_case_indexes)
        test_structure.mf_imputation(k1=10, k2=5, alpha1=0.01, alpha2=0.01, beta1=0.01, beta2=0.01, iteration=3000, verbose='wave')
        assert(test_structure.is_imputed == True)
        print(test_structure.impute_method)
        print(test_structure.estimate_imputation_error(test_case_indexes))
        
        test_structure.reset_imputed()
        test_structure.set_na_values(test_case_indexes)
        test_structure.knn_imputation(k1=5, k2=1, verbose='wave')
        assert(test_structure.is_imputed == True)
        print(test_structure.impute_method)
        print(test_structure.estimate_imputation_error(test_case_indexes))

    def test_knn_subject_impute2(self):
        test_df = pd.read_csv('../../output/v3_python/raw2.csv')
        # Focus on just two waves for testing
        test_structure = Structure(test_df, without_binaries)

        test_structure.knn_impute_subject(k=1, reset=True)

        print(np.isnan(test_structure.imputed).any())
        
    def test_knn_choose_k(self):
        df = pd.read_csv('../../output/v3_python/raw2.csv', index_col=0)

        sys.stdout = open('../../output/logs/compare_impute.log', 'w')

        # Focus on just two waves for testing
        print("========Testing on variables: ")
        print(without_binaries)
        
        data_structure = Structure(df, without_binaries)

        n_samples = int(np.sum(~np.isnan(data_structure.ori)) * (1/100))
        print("========Testing on "+ str(n_samples) + " samples")
                
        print("====Testing for KNN====")
        # Impute waves and calculate the mse for mf
        print("====Testing for k=" + str(25) + '====')
        knn_mses = []
        for i in range(100):
            test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
            data_structure.set_na_values(test_indexes)
            data_structure.knn_impute_wave(k=25, reset=False)
            knn_mses.append(data_structure.estimate_imputation_error(test_indexes))
            data_structure.reset_imputed()
        print('mean_mse='+ str(np.mean(knn_mses)))
        print('mean_stderr='+ str(np.std(knn_mses)/len(knn_mses)))


        print("====Testing for Simple mean imputation====")
        simple_mses=[]
        for i in range(100):
            test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
            data_structure.set_na_values(test_indexes)
            data_structure.simple_impute_wave(reset=False)
            simple_mses.append(data_structure.estimate_imputation_error(test_indexes))
            data_structure.reset_imputed()

        print('mean_mse='+ str(np.mean(simple_mses)))
        print('mean_stderr='+ str(np.std(simple_mses)/len(simple_mses)))

        print("====Testing for multiple imputation BayesianRidge====")
        bayes_ridge_mses=[]
        for i in range(100):
            test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
            data_structure.set_na_values(test_indexes)
            data_structure.multiple_impute_wave(max_iter=100,reset=False)
            bayes_ridge_mses.append(data_structure.estimate_imputation_error(test_indexes))
            data_structure.reset_imputed()
        print('mean_mse='+ str(np.mean(bayes_ridge_mses)))
        print('mean_stderr='+ str(np.std(bayes_ridge_mses)/len(bayes_ridge_mses)))

        print("====Testing for multiple imputation Random Forest====")
        random_forest_mses=[]
        for i in range(100):
            test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
            data_structure.set_na_values(test_indexes)
            data_structure.multiple_impute_wave(estimator='Forest', max_iter=100,reset=False)
            random_forest_mses.append(data_structure.estimate_imputation_error(test_indexes))
            data_structure.reset_imputed()
        print('mean_mse='+ str(np.mean(random_forest_mses)))
        print('mean_stderr='+ str(np.std(random_forest_mses)/len(random_forest_mses)))
            

        print("====Testing for multiple imputation Ridge====")
        ridge_mses=[]
        for i in range(100):
            test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)            
            data_structure.set_na_values(test_indexes)
            data_structure.multiple_impute_wave(estimator='Ridge',max_iter=100,reset=False)
            ridge_mses.append(data_structure.estimate_imputation_error(test_indexes))
            data_structure.reset_imputed()
        print('mean_mse='+ str(np.mean(ridge_mses)))
        print('mean_stderr='+ str(np.std(ridge_mses)/len(ridge_mses)))

        sys.stdout.close()
        
    def test_knn_subject_v2(self):
        df = pd.read_csv('../../output/v3_python/raw2.csv', index_col=0)

        # Focus on just two waves for testing
        print("========Testing on variables: ")
        print(without_binaries)
        
        data_structure = Structure(df, without_binaries)

        data_structure.knn_impute_subject(k=1)

    def test_number_of_NAs(self):
        df = pd.read_csv('../../output/v3_python/raw2.csv', index_col=0)

        # Focus on just two waves for testing
        data_structure = Structure(df, without_binaries)
        n_nas = np.sum(np.isnan(data_structure.imputed))
        print(n_nas)
        m,n,l = data_structure.imputed.shape
        print(m * n * l)
        print(n_nas/(m * n * l))