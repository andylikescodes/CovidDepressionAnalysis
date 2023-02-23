from imputation_data_structure import *
from constants import *
import sys

def impute_cvd(k, columns, percentage_missing):
    df = pd.read_csv('../../output/v3_python/cvd.csv', index_col=0)
    # Focus on just two waves for testing
    print("========Testing on variables: ")
    print(columns)
    
    data_structure = Structure(df, columns)

    n_samples = int(np.sum(~np.isnan(data_structure.ori)) * (percentage_missing/100))
    print("========Testing on "+ str(n_samples) + " samples")
    
    test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
    
    # Impute wave and calculate the mse for knn
    data_structure.reset_imputed()
    data_structure.set_na_values(test_indexes)
    data_structure.knn_impute_wave(k=k, reset=False)
    print('mse='+str(data_structure.estimate_imputation_error(test_indexes)))
    

if __name__ == '__main__':
    k = int(sys.argv[1])
    percentage_missing = int(sys.argv[2])
    
    impute_cvd(k, columns=continuous_covid, percentage_missing=percentage_missing)