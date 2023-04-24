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


def doubly_robust_estimation_cv():
    significant_subjects = []
    for i in range(1, 17):

        logging.info('Started')
        logging.info('====wave '+ str(i))

        # Handle the data
        df = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)))

        T = df['Mandatory_SAH'].values
        y = df['BDI'].values
        X = df.loc[:,[  "Race_A", 
                        "Race_AA", 
                        "Race_W", 
                        "Gender",
                        "Political_Views", 
                        "Age", 
                        "Income",
                        "Education",
                        "NEO_Neuroticism", 
                        "NEO_Extraversion", 
                        "NEO_Openness", 
                        "NEO_Agreeableness", 
                        "NEO_Conscientiousness"]].values
        
        W = df.loc[:,   ['cases_avg_per_100k',
                        'deaths_avg_per_100k',
                        'cases_avg',
                        'deaths_avg',
                        'population']].values

        # CV for the best model
        model_y = clone(first_stage_reg().fit(X, y).best_estimator_)
        print(model_y)

        model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
        print(model_t)

        est = ForestDRLearner(model_regression=model_y,
                            model_propensity=model_t,
                            cv=3,
                            n_estimators=4000,
                            min_samples_leaf=10,
                            verbose=0,
                            min_weight_fraction_leaf=.005)
        est.fit(y, T, X=X, W=W)

        point = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        
        print((lb > 0).any())
        print(np.sum(lb > 0))
        
        if np.sum(lb>0) > 0:
            significant_subjects.append(np.where(lb>0)[0])
    print(np.concatenate(significant_subjects))


def doubly_robust_estimation():
    logging.info('Started')
    significant_subjects = []
    for i in range(1, 17):
        logging.info('====wave '+ str(i))
        df = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)))

        features = [  "Race_A", 
                        "Race_AA", 
                        "Race_W", 
                        "Gender",
                        "Political_Views", 
                        "Age", 
                        "Income",
                        "Education",
                        "NEO_Neuroticism", 
                        "NEO_Extraversion", 
                        "NEO_Openness", 
                        "NEO_Agreeableness", 
                        "NEO_Conscientiousness"]

        controls =  ['cases_avg_per_100k',
                        'deaths_avg_per_100k',
                        'cases_avg',
                        'deaths_avg',
                        'population']

        T = df['Mandatory_SAH'].values
        y = df['BDI'].values
        print(type(T[0]))
        X = df.loc[:, features].values
        
        W = df.loc[:, controls].values
        
        est = ForestDRLearner(model_regression=GradientBoostingRegressor(n_estimators=1000),
                            model_propensity=GradientBoostingClassifier(n_estimators=1000))
        est.fit(y, T, inference='blb', X=X, W=W)
        cate = est.effect(X)
        h_lb, h_ub = est.effect_interval(X, alpha=0.05)

        ate = est.ate(X)
        ate_lb, ate_ub = est.ate_interval(X, alpha=0.05)

        if np.sum(h_lb > 0) > 0:
            significant_subjects.append(np.where(h_lb>0)[0])

        logging.info('Estimated ate='+ str(ate) + ", CI=[{},{}]".format(str(ate_lb),str(ate_ub)) + ", N significance: "+ str(np.sum(h_lb > 0)))

    logging.info(np.concatenate(significant_subjects))


def doubly_robust_estimation_linear():
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started')
    significant_subjects = {}
    for i in range(4, 15):
        logging.info('====wave '+ str(i))
        df = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)))
        df_previous = pd.read_csv('../output/jasp/wave{}.csv'.format(str(i-1)))

        features = ["Race_A", 
                    "Race_AA", 
                    "Race_W", 
                    "Gender",
                    "Political_Views", 
                    "Age", 
                    "Income",
                    "Education",
                    "NEO_Neuroticism", 
                    "NEO_Extraversion", 
                    "NEO_Openness", 
                    "NEO_Agreeableness", 
                    "NEO_Conscientiousness"]

        control_1 =  ['cases_avg_per_100k',
                    'deaths_avg_per_100k',
                    'cases_avg',
                    'deaths_avg',
                    'population']
        
        control_2 = ["lat",
                     "lng"]

        
        T = df_previous['Mandatory_SAH'].values
        y = df['BDI'].values
        X = df.loc[:, features].values
        
        W_1 = df.loc[:, control_1].values
        pca.fit(W_1)
        W_1 = pca.transform(W_1)
        logging.info('2 PC variance explained for W1: {},{}'.format(str(pca.explained_variance_ratio_[0]),str(pca.explained_variance_ratio_[1])))
        #W = poly.fit_transform(W)

        W_2 = df.loc[:, control_2].values

        W = np.hstack([W_1, W_2])

        # CV for the best model
        model_y = clone(first_stage_reg().fit(X, y).best_estimator_)
        logging.info(model_y)

        model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
        logging.info(model_t)

        est = ForestDRLearner(model_regression=model_y,
                            model_propensity=model_t)
        est.fit(y, T, inference='blb', X=X, W=W)
        
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
        fig.savefig('../../output/figures/shap/one_wave_lag_sah/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/one_wave_lag_sah/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)


def doubly_robust_estimation_longitudinal():
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> one wave lag')
    significant_subjects = {}
    for i in range(5, 15):
        logging.info('====wave '+ str(i))
        df = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)))
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)))

        features = ["Race_A", 
                    "Race_AA", 
                    "Race_W", 
                    "Gender",
                    "Political_Views", 
                    "Age", 
                    "Income",
                    "Education",
                    "NEO_Neuroticism", 
                    "NEO_Extraversion", 
                    "NEO_Openness", 
                    "NEO_Agreeableness", 
                    "NEO_Conscientiousness"]

        control_1 =  ['cases_avg_per_100k',
                    'deaths_avg_per_100k',
                    'cases_avg',
                    'deaths_avg',
                    'population']
        
        control_2 = ["lat",
                     "lng"]
        
        control_2_previous = ["NEO_Neuroticism", 
                    "NEO_Extraversion", 
                    "NEO_Openness", 
                    "NEO_Agreeableness", 
                    "NEO_Conscientiousness",
                    "BDI",
                    "STAI",
                    "PSS",
                    "Emot_Support",
                    "Loneliness",
                    "Income",
                    "Political_Views"]

        
        T = df_previous['Mandatory_SAH'].values
        y = df['BDI'].values
        X = df.loc[:, features].values
        
        W_1 = df.loc[:, control_1].values
        pca.fit(W_1)
        W_1 = pca.transform(W_1)
        logging.info('2 PC variance explained for W1: {},{}'.format(str(pca.explained_variance_ratio_[0]),str(pca.explained_variance_ratio_[1])))
        #W = poly.fit_transform(W)

        W_1_previous = df_previous.loc[:, control_2].values
        pca.fit(W_1_previous)
        W_1_previous = pca.transform(W_1_previous)

        W_2 = df.loc[:, control_2].values

        W_2_previous = df_previous.loc[:, control_2_previous].values


        W = np.hstack([W_1, W_1_previous, W_2, W_2_previous])

        # CV for the best model
        model_y = clone(first_stage_reg().fit(X, y).best_estimator_)
        logging.info(model_y)

        model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
        logging.info(model_t)

        est = ForestDRLearner(model_regression=model_y,
                            model_propensity=model_t)
        est.fit(y, T, inference='blb', X=X, W=W)
        
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
        fig.savefig('../../output/figures/shap/one_wave_lag_sah/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/one_wave_lag_sah/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)


def doubly_robust_estimation_longitudinal_previous_confounds():
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> previous confounds')
    significant_subjects = {}
    for i in range(6, 15):
        logging.info('====wave '+ str(i))
        df = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)))
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)))

        features = ["Race_A", 
                    "Race_AA", 
                    "Race_W", 
                    "Gender",
                    "Political_Views", 
                    "Age", 
                    "Income",
                    "Education",
                    "NEO_Neuroticism", 
                    "NEO_Extraversion", 
                    "NEO_Openness", 
                    "NEO_Agreeableness", 
                    "NEO_Conscientiousness"]

        control_covid_severity =  ['cases_avg_per_100k',
                    'deaths_avg_per_100k',
                    'cases_avg',
                    'deaths_avg',
                    'population']
        
        control_location = ["lat",
                     "lng"]

        
        T = df_previous['Mandatory_SAH'].values
        y = df['BDI'].values
        X = df.loc[:, features].values
        
        W_curr_covid_severity = df.loc[:, control_covid_severity].values
        pca.fit(W_curr_covid_severity)
        W_curr_covid_severity = pca.transform(W_curr_covid_severity)
        logging.info('2 PC variance explained for W1: {},{}'.format(str(pca.explained_variance_ratio_[0]),str(pca.explained_variance_ratio_[1])))
        #W = poly.fit_transform(W)

        W_pre_covid_severity = df_previous.loc[:, control_covid_severity].values
        pca.fit(W_pre_covid_severity)
        W_pre_covid_severity = pca.transform(W_pre_covid_severity)

        W_location = df.loc[:, control_location].values


        W = np.hstack([W_curr_covid_severity, W_pre_covid_severity, W_location])

        # CV for the best model
        model_y = clone(first_stage_reg().fit(X, y).best_estimator_)
        logging.info(model_y)

        model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
        logging.info(model_t)

        est = ForestDRLearner(model_regression=model_y,
                            model_propensity=model_t)
        est.fit(y, T, inference='blb', X=X, W=W)
        
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
        fig.savefig('../../output/figures/shap/longitudinal_pre_confounds/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/longitudinal_pre_confounds/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)


def doubly_robust_estimation_longitudinal_subset_age():
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> previous confounds + Age subset')
    significant_subjects = {}
    for i in range(6, 15):
        logging.info('====wave '+ str(i))
        df = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)))
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)))
        df = df.loc[df['Age'] >= df['Age'].mean(),:]
        df_previous = df_previous.loc[df_previous['Age'] >= df_previous['Age'].mean(),:]

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
                    'deaths_avg',
                    'population']
        
        control_location = ["lat",
                     "lng"]

        
        T = df_previous['Mandatory_SAH'].values
        y = df['BDI'].values
        X = df.loc[:, features].values
        
        W_curr_covid_severity = df.loc[:, control_covid_severity].values
        pca.fit(W_curr_covid_severity)
        W_curr_covid_severity = pca.transform(W_curr_covid_severity)
        logging.info('2 PC variance explained for W1: {},{}'.format(str(pca.explained_variance_ratio_[0]),str(pca.explained_variance_ratio_[1])))
        #W = poly.fit_transform(W)

        W_pre_covid_severity = df_previous.loc[:, control_covid_severity].values
        pca.fit(W_pre_covid_severity)
        W_pre_covid_severity = pca.transform(W_pre_covid_severity)

        W_location = df.loc[:, control_location].values


        W = np.hstack([W_curr_covid_severity, W_pre_covid_severity, W_location])

        # CV for the best model
        model_y = clone(first_stage_reg().fit(X, y).best_estimator_)
        logging.info(model_y)

        model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
        logging.info(model_t)

        est = ForestDRLearner(model_regression=model_y,
                            model_propensity=model_t)
        est.fit(y, T, inference='blb', X=X, W=W)
        
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
        fig.savefig('../../output/figures/shap/age_subset/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/age_subset/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)


def doubly_robust_estimation_longitudinal_subset_neuroticism():
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> previous confounds + neuroticism subset')
    significant_subjects = {}
    for i in range(6, 15):
        logging.info('====wave '+ str(i))
        df = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)))
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)))
        df_curr = df.loc[df['NEO_Neuroticism'] >= df['NEO_Neuroticism'].mean(),:]
        df_previous = df_previous.loc[df['NEO_Neuroticism'] >= df['NEO_Neuroticism'].mean(),:]

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
                    'deaths_avg',
                    'population']
        
        control_location = ["lat",
                     "lng"]

        
        T = df_previous['Mandatory_SAH'].values
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


        W = np.hstack([W_pre_covid_severity, W_location])

        # CV for the best model
        model_y = clone(first_stage_reg().fit(X, y).best_estimator_)
        logging.info(model_y)

        model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
        logging.info(model_t)

        est = ForestDRLearner(model_regression=model_y,
                            model_propensity=model_t)
        est.fit(y, T, inference='blb', X=X, W=W)
        
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
        fig.savefig('../../output/figures/shap/neuroticism_subset/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/neuroticism_subset/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)


def doubly_robust_estimation_longitudinal_subset_neuroticism_age():
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> previous confounds + neuroticism subset + age subset')
    significant_subjects = {}
    for i in range(6, 15):
        logging.info('====wave '+ str(i))
        df = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)))
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)))
        df_curr = df.loc[(df['NEO_Neuroticism'] >= df['NEO_Neuroticism'].mean()) & (df['Age'] >= df['Age'].mean()),:]
        df_previous = df_previous.loc[(df['NEO_Neuroticism'] >= df['NEO_Neuroticism'].mean()) & (df['Age'] >= df['Age'].mean()),:]
        print(df_curr.shape)

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
                    'deaths_avg',
                    'population']
        
        control_location = ["lat",
                     "lng"]

        
        T = df_previous['Mandatory_SAH'].values
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


        W = np.hstack([W_pre_covid_severity, W_location])

        # CV for the best model
        model_y = clone(first_stage_reg().fit(X, y).best_estimator_)
        logging.info(model_y)

        model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
        logging.info(model_t)

        est = ForestDRLearner(model_regression=model_y,
                            model_propensity=model_t)
        est.fit(y, T, inference='blb', X=X, W=W)
        
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
        fig.savefig('../../output/figures/shap/neuroticism_age_subset/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/neuroticism_age_subset/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)


def doubly_robust_estimation_longitudinal_subset_neuroticism_extraversion_age():
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> previous confounds + neuroticism subset + age subset + extraversion')
    significant_subjects = {}
    for i in range(6, 15):
        logging.info('====wave '+ str(i))
        df = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)))
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)))
        df_curr = df.loc[(df['NEO_Neuroticism'] >= df['NEO_Neuroticism'].mean()) & (df['Age'] >= df['Age'].mean()) & (df['NEO_Extraversion'] <= df['NEO_Extraversion'].mean()),:]
        df_previous = df_previous.loc[(df['NEO_Neuroticism'] >= df['NEO_Neuroticism'].mean()) & (df['Age'] >= df['Age'].mean()) & (df['NEO_Extraversion'] <= df['NEO_Extraversion'].mean()),:]
        print(df_curr.shape)

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
                    'deaths_avg',
                    'population']
        
        control_location = ["lat",
                     "lng"]

        
        T = df_previous['Mandatory_SAH'].values
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


        W = np.hstack([W_pre_covid_severity, W_location])

        # CV for the best model
        model_y = clone(first_stage_reg().fit(X, y).best_estimator_)
        logging.info(model_y)

        model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
        logging.info(model_t)

        est = ForestDRLearner(model_regression=model_y,
                            model_propensity=model_t)
        est.fit(y, T, inference='blb', X=X, W=W)
        
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
        fig.savefig('../../output/figures/shap/neuroticism_age_extraversion_subset/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/neuroticism_age_extraversion_subset/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)


def doubly_robust_estimation_longitudinal_subset_neuroticism_extraversion_age_conscientiousness():
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> previous confounds + neuroticism subset + age subset + extraversion + conscientiousness')
    significant_subjects = {}
    for i in range(6, 15):
        logging.info('====wave '+ str(i))
        df = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)))
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)))
        df_curr = df.loc[(df['NEO_Neuroticism'] >= df['NEO_Neuroticism'].mean()) & (df['Age'] >= df['Age'].mean()) & (df['NEO_Extraversion'] <= df['NEO_Extraversion'].mean()) & (df['NEO_Conscientiousness'] >= df['NEO_Conscientiousness'].mean()),:]
        df_previous = df_previous.loc[(df['NEO_Neuroticism'] >= df['NEO_Neuroticism'].mean()) & (df['Age'] >= df['Age'].mean()) & (df['NEO_Extraversion'] <= df['NEO_Extraversion'].mean()) & (df['NEO_Conscientiousness'] >= df['NEO_Conscientiousness'].mean()),:]
        print(df_curr.shape)

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
                    'deaths_avg',
                    'population']
        
        control_location = ["lat",
                     "lng"]

        
        T = df_previous['Mandatory_SAH'].values
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


        W = np.hstack([W_pre_covid_severity, W_location])

        # CV for the best model
        model_y = clone(first_stage_reg().fit(X, y).best_estimator_)
        logging.info(model_y)

        model_t = clone(first_stage_clf().fit(X, T).best_estimator_)
        logging.info(model_t)

        est = ForestDRLearner(model_regression=model_y,
                            model_propensity=model_t)
        est.fit(y, T, inference='blb', X=X, W=W)
        
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
        fig.savefig('../../output/figures/shap/neuroticism_age_extraversion_conscientiousness_subset/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/neuroticism_age_extraversion_conscientiousness_subset/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)


def check_significant_subjects():
    with open('../../output/v3_python/significant_subjects.pickle', 'rb') as f:
        subjects = pkl.load(f)

    all_subjects = []
    for i in subjects.keys():
        for j in range(len(subjects[i])):
            all_subjects.append(subjects[i][j])

    print(np.unique(all_subjects))

def doubly_robust_estimation_positive_corr():

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("../../output/logs/doubly_robust_estimation_selected_samples_positive_corr.log"),
            logging.StreamHandler()
        ]
    )

    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> positive corr selected samples')
    significant_subjects = {}

    analysis_tool = LongitudinalAnalysis()

    positive_indexes, negative_indexes = analysis_tool.extract_significant_subjects(analysis_tool.bdi_sah_corr())

    for i in range(2, 17):
        logging.info('====wave '+ str(i))
        df_curr = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)), index_col=0)
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)), index_col=0)

        # Selecting subjects
        logging.info('Number of positive subjects:' + str(np.sum(positive_indexes)))
        df_curr = df_curr.loc[positive_indexes, :]
        df_previous = df_previous.loc[positive_indexes, :]


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

        if np.sum(T) == len(T):
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
        fig.savefig('../../output/figures/shap/longitudinal_reanalysis/positive_corr_subset/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/longitudinal_reanalysis/positive_corr_subset/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)


def doubly_robust_estimation_negative_corr():

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("../../output/logs/doubly_robust_estimation_selected_samples_negative_corr.log"),
            logging.StreamHandler()
        ]
    )

    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> negative corr selected samples')
    significant_subjects = {}

    analysis_tool = LongitudinalAnalysis()

    positive_indexes, negative_indexes = analysis_tool.extract_significant_subjects(analysis_tool.bdi_sah_corr())

    for i in range(2, 17):
        logging.info('====wave '+ str(i))
        df_curr = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)), index_col=0)
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)), index_col=0)

        # Selecting subjects
        logging.info('Number of negative subjects:' + str(np.sum(negative_indexes)))
        df_curr = df_curr.loc[negative_indexes, :]
        df_previous = df_previous.loc[negative_indexes, :]


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

        if np.sum(T) == len(T):
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
        fig.savefig('../../output/figures/shap/longitudinal_reanalysis/negative_corr_subset/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/longitudinal_reanalysis/negative_corr_subset/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)

def doubly_robust_estimation_generic(type):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("../../output/logs/{}.log".format(type)),
            logging.StreamHandler()
        ]
    )

    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> {} selected samples'.format(type))
    significant_subjects = {}

    analysis_tool = LongitudinalAnalysis()

    positive_indexes, negative_indexes = analysis_tool.extract_significant_subjects(analysis_tool.bdi_sah_corr())

    selected_indexes = None

    if type == 'Negative':
        selected_indexes = negative_indexes
    elif type == 'Positive':
        selected_indexes = positive_indexes
    elif type == 'Neutral':
        selected_indexes = ~(negative_indexes | positive_indexes)

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

        if np.sum(T) == len(T):
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
        fig.savefig('../../output/figures/shap/longitudinal_reanalysis/{}/wave{}.svg'.format(type, str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/longitudinal_reanalysis/significant_subjects_{}.pickle'.format(type), 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)

def doubly_robust_estimation_neutral_corr():

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("../../output/logs/doubly_robust_estimation_selected_samples_neutral_corr.log"),
            logging.StreamHandler()
        ]
    )

    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    pca = PCA(n_components=2)
    logging.info('Started -> neutral corr selected samples')
    significant_subjects = {}

    analysis_tool = LongitudinalAnalysis()

    positive_indexes, negative_indexes = analysis_tool.extract_significant_subjects(analysis_tool.bdi_sah_corr())

    for i in range(2, 17):
        logging.info('====wave '+ str(i))
        df_curr = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i)), index_col=0)
        df_previous = pd.read_csv('../../output/jasp/wave{}.csv'.format(str(i-1)), index_col=0)

        # Selecting subjects
        indexes = ~(negative_indexes | positive_indexes)
        logging.info('Number of neutral subjects:' + str(np.sum(indexes)))
        df_curr = df_curr.loc[indexes, :]
        df_previous = df_previous.loc[indexes, :]


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

        if np.sum(T) == len(T):
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
        fig.savefig('../../output/figures/shap/longitudinal_reanalysis/neutral_corr_subset/wave{}.svg'.format(str(i)), bbox_inches='tight')
        plt.clf()
    
    with open('../../output/v3_python/longitudinal_reanalysis/neutral_corr_subset/significant_subjects.pickle', 'wb') as handle:
        pkl.dump(significant_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    doubly_robust_estimation_positive_corr()
    doubly_robust_estimation_negative_corr()
    doubly_robust_estimation_neutral_corr()