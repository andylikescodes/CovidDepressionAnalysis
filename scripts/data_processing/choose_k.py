from imputation_data_structure import *
import sys

def impute_full(k1, k2):
    test_df = pd.read_csv('../../output/v3_python/raw.csv')
    # Focus on just two waves for testing
    data_structure = Structure(test_df)
    data_structure.run_imputation(k1=k1, k2=k2, alpha1=0.01, alpha2=0.01, beta1=0.01, beta2=0.01, iteration=3000)


if __name__ == '__main__':
    print(sys.argv[1])
    k1 = int(sys.argv[1])
    k2 = int(sys.argv[2])
    impute_full(k1, k2)