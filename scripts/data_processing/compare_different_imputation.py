from imputation_data_structure import *
import pandas as pd
import math
from random import sample, choices
from constants import *
import sys
from tqdm import tqdm
import logging


def compare_different_imputation_wave(columns):
    df = pd.read_csv('../../output/v3_python/raw2.csv', index_col=0)

    #sys.stdout = open('../../output/logs/compare_different_imputation.log', 'w')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("../../output/logs/compare_imputation.log"),
            logging.StreamHandler()
        ]
    )
    logging.info('Started')
    # Focus on just two waves for testing
    logging.info("========Testing on variables: ")
    logging.info(columns)
    
    data_structure = Structure(df, columns)

    n_samples = int(np.sum(~np.isnan(data_structure.ori)) * (0.5/100))
    logging.info("========Testing on "+ str(n_samples) + " samples")
            
    logging.info("====Testing for KNN====")
    # Impute waves and calculate the mse for mf

    for k in [1,3,5,10,15,20,25]:
        logging.info("====Testing for k=" + str(k) + '====')
        knn_mses = []
        for i in tqdm(range(100)):
            test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
            data_structure.set_na_values(test_indexes)
            data_structure.knn_impute_wave(k=k, reset=False)
            mse, valid_testcase = data_structure.estimate_imputation_error(test_indexes)
            if i % 10 == 0:
                logging.info('iter='+str(i)+' mse='+str(mse))
            knn_mses.append(mse)
            data_structure.reset_imputed()
        logging.info('mean_mse='+ str(np.mean(knn_mses)))
        logging.info('mean_stderr='+ str(np.std(knn_mses)/np.sqrt(len(knn_mses)-1)))
        logging.info('valid_testcase='+str(valid_testcase))


    logging.info("====Testing for Simple mean imputation====")
    simple_mses=[]
    for i in tqdm(range(100)):
        test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
        data_structure.set_na_values(test_indexes)
        data_structure.simple_impute_wave(reset=False)
        mse, valid_testcase = data_structure.estimate_imputation_error(test_indexes)
        if i % 10 == 0:
            logging.info('iter='+str(i)+' mse='+str(mse))
        simple_mses.append(mse)
        data_structure.reset_imputed()

    logging.info('mean_mse='+ str(np.mean(simple_mses)))
    logging.info('mean_stderr='+ str(np.std(simple_mses)/np.sqrt(len(simple_mses)-1)))

    logging.info("====Testing for multiple imputation BayesianRidge====")
    bayes_ridge_mses=[]
    for i in tqdm(range(100)):
        test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
        data_structure.set_na_values(test_indexes)
        data_structure.multiple_impute_wave(max_iter=100,reset=False)
        mse, valid_testcase = data_structure.estimate_imputation_error(test_indexes)
        if i % 10 == 0:
            logging.info('iter='+str(i)+' mse='+str(mse))
        bayes_ridge_mses.append(mse)
        data_structure.reset_imputed()
    logging.info('mean_mse='+ str(np.mean(bayes_ridge_mses)))
    logging.info('mean_stderr='+ str(np.std(bayes_ridge_mses)/np.sqrt(len(bayes_ridge_mses)-1)))

    logging.info("====Testing for multiple imputation Ridge====")
    ridge_mses=[]
    for i in tqdm(range(100)):
        test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)            
        data_structure.set_na_values(test_indexes)
        data_structure.multiple_impute_wave(estimator='Ridge',max_iter=100,reset=False)
        mse, valid_testcase = data_structure.estimate_imputation_error(test_indexes)
        if i % 10 == 0:
            logging.info('iter='+str(i)+' mse='+str(mse))
        ridge_mses.append(mse)
        data_structure.reset_imputed()
    logging.info('mean_mse='+ str(np.mean(ridge_mses)))
    logging.info('mean_stderr='+ str(np.std(ridge_mses)/np.sqrt(len(ridge_mses)-1)))

    logging.info("====Testing for MF k=10====")
    MF_mses = []
    for i in tqdm(range(100)):
        test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)            
        data_structure.set_na_values(test_indexes)
        data_structure.mf_impute_wave(k=10, iter=1500,reset=False)
        if i % 10 == 0:
            logging.info('iter='+str(i)+' mse='+str(mse))
        logging.info('iter='+str(i)+' mse='+str(mse))
        MF_mses.append(mse)
        data_structure.reset_imputed()
    logging.info('mean_mse='+ str(np.mean(MF_mses)))
    logging.info('mean_stderr='+ str(np.std(MF_mses)/np.sqrt(len(MF_mses)-1)))

    logging.info("====Testing for multiple imputation Random Forest====")
    random_forest_mses=[]
    for i in tqdm(range(100)):
        test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
        data_structure.set_na_values(test_indexes)
        data_structure.multiple_impute_wave(estimator='Forest', max_iter=100,reset=False)
        mse = data_structure.estimate_imputation_error(test_indexes)
        if i % 10 == 0:
            logging.info('iter='+str(i)+' mse='+str(mse))
        random_forest_mses.append(mse)
        data_structure.reset_imputed()
    logging.info('mean_mse='+ str(np.mean(random_forest_mses)))
    logging.info('mean_stderr='+ str(np.std(random_forest_mses)/np.sqrt(len(random_forest_mses)-1)))

    #sys.stdout.close()
    logging.info('Finished')

def get_targeted_indexes(test_indexes, data_structure, selected_vars):
    indexes = []
    for var in selected_vars:
        indexes.append(data_structure.meta['columns'].index(var))

    selected_indexes = []
    for idx in test_indexes:
        if idx[1] in indexes:
            selected_indexes.append(idx)
    return selected_indexes


def compare_different_imputation_subject(columns):
    df = pd.read_csv('../../output/v3_python/raw2.csv', index_col=0)

    target_vars = ['BDI', 'PSS', 'STAI', 'Fear', 'Emot_Support', 'Loneliness']

    #sys.stdout = open('../../output/logs/compare_different_imputation.log', 'w')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("../../output/logs/compare_imputation_subject.log"),
            logging.StreamHandler()
        ]
    )
    logging.info('Started')
    # Focus on just two waves for testing
    logging.info("========Testing on variables: ")
    logging.info(columns)
    logging.info("targeted variables:")
    logging.info(target_vars)
    
    data_structure = Structure(df, columns)

    n_samples = int(np.sum(~np.isnan(data_structure.ori)) * (10/100))
    logging.info("========Testing on "+ str(n_samples) + " samples")
            
    logging.info("====Testing for KNN====")
    # Impute waves and calculate the mse for mf

    for k in [1,2,3]:
        logging.info("====Testing for k=" + str(k) + '====')
        knn_mses = []
        knn_target_mses = []
        for i in tqdm(range(100)):
            test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
            target_indexes = get_targeted_indexes(test_indexes, data_structure, target_vars)
            data_structure.set_na_values(test_indexes)
            data_structure.knn_impute_subject(k=k, reset=False)
            mse, valid_test = data_structure.estimate_imputation_error(test_indexes)
            target_mse, target_valid_test = data_structure.estimate_imputation_error(target_indexes)
            if i % 10 == 0:
                logging.info('iter='+str(i)+' full_mse='+str(mse) + ' valid_testcase='+str(valid_test))
                logging.info('target_mse='+str(target_mse) + ' valid_target_testcase='+str(target_valid_test))
            knn_mses.append(mse)
            knn_target_mses.append(target_mse)
            data_structure.reset_imputed()
        logging.info('mean_mse='+ str(np.mean(knn_mses)))
        logging.info('mean_stderr='+ str(np.std(knn_mses)/np.sqrt(len(knn_mses)-1)))
        logging.info('mean_target_mse='+ str(np.mean(knn_target_mses)))
        logging.info('mean_target_stderr='+ str(np.std(knn_target_mses)/np.sqrt(len(knn_target_mses)-1)))

    logging.info("====Testing for Simple mean imputation====")
    simple_mses=[]
    simple_target_mses=[]
    for i in tqdm(range(100)):
        test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
        target_indexes = get_targeted_indexes(test_indexes, data_structure, target_vars)
        data_structure.set_na_values(test_indexes)
        data_structure.simple_impute_subject(reset=False)
        mse, valid_test = data_structure.estimate_imputation_error(test_indexes)
        target_mse, target_valid_test = data_structure.estimate_imputation_error(target_indexes)
        if i % 10 == 0:
            logging.info('iter='+str(i)+' full_mse='+str(mse) + ' valid_testcase='+str(valid_test))
            logging.info('target_mse='+str(target_mse) + ' valid_target_testcase='+str(target_valid_test))
        simple_mses.append(mse)
        simple_target_mses.append(target_mse)
        data_structure.reset_imputed()

    logging.info('mean_mse='+ str(np.mean(simple_mses)))
    logging.info('mean_stderr='+ str(np.std(simple_mses)/np.sqrt(len(simple_mses)-1)))
    logging.info('mean_target_mse='+ str(np.mean(simple_target_mses)))
    logging.info('mean_target_stderr='+ str(np.std(simple_target_mses)/np.sqrt(len(simple_target_mses)-1)))
    

    logging.info("====Testing for multiple imputation BayesianRidge====")
    bayes_ridge_mses=[]
    bayes_ridge_target_mses=[]
    for i in tqdm(range(100)):
        test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)
        target_indexes = get_targeted_indexes(test_indexes, data_structure, target_vars)
        data_structure.set_na_values(test_indexes)
        data_structure.multiple_impute_subject(max_iter=100,reset=False)
        mse, valid_test = data_structure.estimate_imputation_error(test_indexes)
        target_mse, target_valid_test = data_structure.estimate_imputation_error(target_indexes)
        if i % 10 == 0:
            logging.info('iter='+str(i)+' full_mse='+str(mse) + ' valid_testcase='+str(valid_test))
            logging.info('target_mse='+str(target_mse) + ' valid_target_testcase='+str(target_valid_test))
        bayes_ridge_mses.append(mse)
        bayes_ridge_target_mses.append(target_mse)
        data_structure.reset_imputed()
    logging.info('mean_mse='+ str(np.mean(bayes_ridge_mses)))
    logging.info('mean_stderr='+ str(np.std(bayes_ridge_mses)/np.sqrt(len(bayes_ridge_mses)-1)))
    logging.info('mean_target_mse='+ str(np.mean(bayes_ridge_target_mses)))
    logging.info('mean_target_stderr='+ str(np.std(bayes_ridge_target_mses)/np.sqrt(len(bayes_ridge_target_mses)-1)))

    # logging.info("====Testing for MF k=5====")
    # MF_mses = []
    # MF_target_mses=[]
    # for i in tqdm(range(100)):
    #     test_indexes, _ = data_structure.create_testing_data(n_samples=n_samples)   
    #     target_indexes = get_targeted_indexes(test_indexes, data_structure, target_vars)         
    #     data_structure.set_na_values(test_indexes)
    #     data_structure.mf_impute_subject(k=5, iter=1500, reset=False)
    #     mse, valid_test = data_structure.estimate_imputation_error(test_indexes)
    #     target_mse, target_valid_test = data_structure.estimate_imputation_error(target_indexes)
    #     if i % 10 == 0:
    #         logging.info('iter='+str(i)+' full_mse='+str(mse) + ' valid_testcase='+str(valid_test))
    #         logging.info('target_mse='+str(target_mse) + ' valid_target_testcase='+str(target_valid_test))
    #     MF_mses.append(mse)
    #     MF_target_mses.append(target_mse)
    #     data_structure.reset_imputed()
    # logging.info('mean_mse='+ str(np.mean(MF_mses)))
    # logging.info('mean_stderr='+ str(np.std(MF_mses)/len(MF_mses)))
    # logging.info('mean_target_mse='+ str(np.mean(MF_target_mses)))
    # logging.info('mean_target_stderr='+ str(np.std(MF_target_mses)/len(MF_target_mses)))

    #sys.stdout.close()
    logging.info('Finished')

if __name__ == '__main__':
    logging.info("!!!!! Corrected scaling method for imputation")
    compare_different_imputation_subject(without_binaries)