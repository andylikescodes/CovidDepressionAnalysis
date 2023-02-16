from imputation_data_structure import *
import pandas as pd
import math
from random import sample, choices

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
