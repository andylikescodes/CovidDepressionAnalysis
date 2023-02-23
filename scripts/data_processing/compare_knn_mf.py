from imputation_data_structure import *
from constants import *
import sys

def impute_full(mf_k1, mf_k2, knn_k1, knn_k2, iter, columns, percentage_missing):
    df = pd.read_csv('../../output/v3_python/raw2.csv', index_col=0)
    # Focus on just two waves for testing
    print("========Testing on variables: ")
    print(columns)
    
    data_structure = Structure(df, columns)

    n_samples = int(np.sum(~np.isnan(data_structure.ori)) * (percentage_missing/100))
    print("========Testing on "+ str(n_samples) + " samples")
    
    test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
    
    # Impute waves and calculate the mse for mf
    data_structure.reset_imputed()
    data_structure.set_na_values(test_indexes)
    data_structure.mf_impute_wave(k=mf_k1, iter=iter, reset=False, verbose=True)
    print(data_structure.estimate_imputation_error(test_indexes))
    
    # Impute subjects and calculate the mse for mf
    data_structure.reset_imputed()
    data_structure.set_na_values(test_indexes)
    data_structure.mf_impute_subject(k=mf_k2, iter=iter, reset=False, verbose=False)
    print(data_structure.estimate_imputation_error(test_indexes))
    
    # Impute wave and calculate the mse for knn
    data_structure.reset_imputed()
    data_structure.set_na_values(test_indexes)
    data_structure.knn_impute_wave(k=knn_k1, reset=False)
    print(data_structure.estimate_imputation_error(test_indexes))
    
    # Impute subject and calculate the mse for knn
    data_structure.reset_imputed()
    data_structure.set_na_values(test_indexes)
    data_structure.knn_impute_subject(k=knn_k2, reset=False)
    print(data_structure.estimate_imputation_error(test_indexes))    

if __name__ == '__main__':
    mf_k1 = int(sys.argv[1])
    mf_k2 = int(sys.argv[2])
    knn_k1 = int(sys.argv[3])
    knn_k2 = int(sys.argv[4])
    iter = int(sys.argv[5])
    percentage_missing = int(sys.argv[6])
    
    for b in [with_binaries, without_binaries]:
    
        impute_full(mf_k1=mf_k1, mf_k2=mf_k2, knn_k1=knn_k1, knn_k2=knn_k2, iter=iter, columns=b, percentage_missing=percentage_missing)