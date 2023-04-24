from econml.dr import ForestDRLearner
from econml.dr import LinearDRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import shap

from econml.dr import DRLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import sys
import os
import pickle as pkl
# adding  to the system path
sys.path.insert(0, '../data_processing')
sys.path.insert(0, '../individual_level')
from constants import *
from longitudinal_analysis import *

from econml.sklearn_extensions.model_selection import GridSearchCVList
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
from econml.sklearn_extensions.linear_model import WeightedLasso

import logging

def first_stage_reg():
    return GridSearchCVList([RandomForestRegressor(n_estimators=100, random_state=123),
                             GradientBoostingRegressor(random_state=123)],
                             param_grid_list=[{'n_estimators': [50, 100, 150, 200],
                                                'max_depth': [3, 5, 10, None],
                                               'min_samples_leaf': [1, 5, 10, 20, 50, 75, 100]},
                                              {'n_estimators': [50, 100, 150, 200],
                                               'max_depth': [3, 5, 10, None],
                                               'min_samples_leaf': [1, 5, 10, 20, 50, 75, 100]}],
                             cv=5)

def first_stage_clf():
    return GridSearchCVList([RandomForestClassifier(random_state=123),
                             GradientBoostingClassifier(random_state=123)],
                             param_grid_list=[{'n_estimators': [50, 100, 150, 200],
                                               'max_depth': [3, 5, 10, None],
                                               'min_samples_leaf':[1, 5, 10, 20, 50, 75, 100]},
                                              {'n_estimators': [50, 100, 150, 200],
                                               'max_depth': [3, 5, 10, None],
                                               'min_samples_leaf': [1, 5, 10, 20, 50, 75, 100]}],
                             cv=5)

def final_stage():
    return GridSearchCVList([WeightedLasso(),
                             RandomForestRegressor(n_estimators=100, random_state=123)],
                             param_grid_list=[{'alpha': [.001, .01, .1, 1, 10]},
                                              {'max_depth': [3, 5],
                                               'min_samples_leaf': [10, 50]}],
                             cv=5,
                             scoring='neg_mean_squared_error')

def doubly_robust_estimation_generic(type='all'):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("../../output/logs/longitudinal_reanalysis/{}.log".format(type)),
            logging.StreamHandler()
        ]
    )

    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> {} selected samples'.format(type))
    significant_subjects = {}

    analysis_tool = LongitudinalAnalysis()

    positive_indexes, negative_indexes = analysis_tool.extract_significant_subjects(analysis_tool.bdi_sah_corr())

    selected_indexes = np.ones(len(positive_indexes), dtype=bool)

    if type == 'Negative':
        selected_indexes = negative_indexes
    elif type == 'Positive':
        selected_indexes = positive_indexes
    elif type == 'Neutral':
        selected_indexes = ~(negative_indexes | positive_indexes)

    save_path = '../../output/figures/shap/longitudinal_reanalysis/' + type
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in range(2, 17):
        logging.info('====wave '+ str(i))
        df_curr = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)), index_col=0)
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)), index_col=0)

        # Selecting subjects
        logging.info('Number of selected subjects:' + str(np.sum(selected_indexes)))
        df_curr = df_curr.loc[selected_indexes, :]
        df_previous = df_previous.loc[selected_indexes, :]


        features = ["Race_A", 
                    "Race_AA", 
                    "Race_W", 
                    "Gender",
                    "Political_Views", 
                    "Income",
                    "Age",
                    "Education",
                    "NEO_Neuroticism", 
                    "NEO_Extraversion", 
                    "NEO_Openness", 
                    "NEO_Agreeableness", 
                    "NEO_Conscientiousness"]

        control_covid_severity =  ['cases_avg_per_100k',
                    'deaths_avg_per_100k',
                    'cases_avg',
                    'deaths_avg']
        
        control_location = ["lat",
                            "lng",
                            "population"]

        T = df_curr['Mandatory_SAH'].values

        logging.info('number of people unders SAH:' + str(np.sum(T)))

        if (np.sum(T) == len(T) | (np.sum(T) == 0)):
            logging.info('No variance in treatment')
            continue

        y = df_curr['BDI'].values
        X = df_curr.loc[:, features].values
        
        W_curr_covid_severity = df_curr.loc[:, control_covid_severity].values
        pca.fit(W_curr_covid_severity)
        W_curr_covid_severity = pca.transform(W_curr_covid_severity)
        logging.info('2 PC variance explained for W1: {},{}'.format(str(pca.explained_variance_ratio_[0]),str(pca.explained_variance_ratio_[1])))
        #W = poly.fit_transform(W)

        W_pre_covid_severity = df_previous.loc[:, control_covid_severity].values
        pca.fit(W_pre_covid_severity)
        W_pre_covid_severity = pca.transform(W_pre_covid_severity)

        W_location = df_curr.loc[:, control_location].values

        W = np.hstack([W_pre_covid_severity, W_curr_covid_severity, W_location])

        try:
            # CV for the best model
            model_y = clone(first_stage_reg().fit(X, y).best_estimator_)
            logging.info(model_y)

            model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
            logging.info(model_t)

            est = ForestDRLearner(model_regression=model_y,
                                model_propensity=model_t)
        
        
            est.fit(y, T, inference='blb', X=X, W=W)
        except Exception as e:
            logging.info(str(e))
            continue
        
        # est = LinearDRLearner()
        #est.fit(y, T, X=X, W=W)
        cate = est.effect(X)
        h_lb, h_ub = est.effect_interval(X, alpha=0.05)

        ate = est.ate(X)
        ate_lb, ate_ub = est.ate_interval(X, alpha=0.05)

        if np.sum(h_lb > 0) > 0:
            significant_subjects[i] = np.where(h_lb>0)[0]
        else:
            significant_subjects[i] = np.array([])

        logging.info('Estimated ate='+ str(ate) + ", CI=[{},{}]".format(str(ate_lb),str(ate_ub)) + ", N significance: "+ str(np.sum(h_lb > 0)))
        logging.info(est.feature_importances_(T=1))

        fig=plt.gcf()
        shap_values = est.shap_values(X, feature_names=features, background_samples=100)
        shap.summary_plot(shap_values["Y0"]['T0_True'], show=False)
        fig.savefig(save_path + '/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/longitudinal_reanalysis/significant_subjects_{}.pickle'.format(type), 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # doubly_robust_estimation_generic('Positive')
    # doubly_robust_estimation_generic('Negative')
    # doubly_robust_estimation_generic('Neutral')

    doubly_robust_estimation_generic()