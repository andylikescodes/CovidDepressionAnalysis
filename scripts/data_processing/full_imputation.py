from imputation_data_structure import *
from constants import *
import sys
import pickle

def impute_core(k, columns):
    df = pd.read_csv('../../output/v3_python/raw2.csv', index_col=0)
    # Focus on just two waves for testing
    print("========Imputing on variables: ")
    print(columns)
    data_structure = Structure(df, columns)
    # Impute wave and calculate the mse for knn
    data_structure.knn_impute_subject(k=k, reset=True)
    with open('../../output/v3_python/core_imputed.pickle', 'wb') as handle:
        pickle.dump(data_structure, handle, protocol=pickle.HIGHEST_PROTOCOL)

def impute_cvd(k, columns):
    df = pd.read_csv('../../output/v3_python/cvd.csv', index_col=0)
    # Focus on just two waves for testing
    print("========Imputing on variables: ")
    print(columns)
    data_structure = Structure(df, columns)
    # Impute wave and calculate the mse for knn
    data_structure.knn_impute_wave(k=k, reset=True)
    with open('../../output/v3_python/cvd_imputed.pickle', 'wb') as handle:
        pickle.dump(data_structure, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    k1 = int(sys.argv[1])
    k2 = int(sys.argv[2])
    
    impute_core(k1, columns=with_binaries)
    impute_cvd(k2, columns=continuous_covid)