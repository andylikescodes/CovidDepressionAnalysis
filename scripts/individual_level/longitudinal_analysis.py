import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

import pickle as pkl

from scipy import stats
from sklearn.decomposition import PCA
import seaborn as sns

class LongitudinalAnalysis():
    def __init__(self):
        with open('/home/andy/CovidDepressionAnalysis/output/v3_python/full_imputed.pickle', 'rb') as f:
            imputed = pkl.load(f)

        self.data = imputed['data']
        self.columns = imputed['columns']
        self.ids = imputed['cvdids']

        self.index_bdi = imputed['columns'].index('BDI')
        self.index_sah = imputed['columns'].index('Mandatory_SAH')

    def compare_before_after_sah(self, data):
        before_sah = []
        after_sah = []
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                if data[i, self.index_sah, j] == 1:
                    after_sah.append(data[i, self.index_bdi, j])
                else:
                    before_sah.append(data[i, self.index_bdi, j])
        return before_sah, after_sah
    
    def pearson_corr(self, x1, x2):
        return (stats.ttest_ind(x1, x2))
    
    def corr_m(self, m1, m2):
        corr_out = []
        pval_out = []

        for i in range(self.data.shape[0]):
            pearson_corr = stats.pearsonr(m1[i,:], m2[i,:])
            corr_out.append(pearson_corr[0])
            pval_out.append(pearson_corr[1])

        df = pd.DataFrame({'corr':corr_out, 'pval':pval_out})
        return df
    
    def corr(self, var1, var2):
        corr_out = []
        pval_out = []

        index_var1 = self.columns.index(var1)
        index_var2 = self.columns.index(var2)

        for i in range(self.data.shape[0]):
            m_var1 = self.data[i, index_var1, :]
            m_var2 = self.data[i, index_var2, :]
            pearson_corr = stats.pearsonr(m_var1, m_var2)
            corr_out.append(pearson_corr[0])
            pval_out.append(pearson_corr[1])

        df = pd.DataFrame({'corr':corr_out, 'pval':pval_out})
        return df
    
    def bdi_sah_corr(self):
        corr_out = []
        pval_out = []
        for i in range(self.data.shape[0]):
            BDI = self.data[i, self.index_bdi, :]
            SAH = self.data[i, self.index_sah, :]
            pearson_corr = stats.pearsonr(BDI, SAH)
            corr_out.append(pearson_corr[0])
            pval_out.append(pearson_corr[1])

        df = pd.DataFrame({'corr':corr_out, 'pval':pval_out})
        return df
    
    def extract_significant_subjects(self, df):
        significant_positive_indexes = ((df['pval'] < 0.05) & (df['corr'] > 0)).values
        significant_negative_indexes = ((df['pval'] < 0.05) & (df['corr'] < 0)).values
        return significant_positive_indexes, significant_negative_indexes
    

    def get_columns(self, columns):
        indexes = []
        for i in range(len(columns)):
            indexes.append(self.columns.index(columns[i]))

        return self.data[:, indexes,:]
    
    def plot_columns_with_groups(self, columns, groups):
        df = pd.DataFrame(columns=columns, data=np.mean(self.get_columns(columns), 2))
        df['group'] = groups

        df = df.melt(id_vars=['group'], value_vars=columns)

        fig = plt.figure(figsize = (20, 10))

        sns.barplot(data=df, x='variable', y='value', hue='group')
        
    def subject_classification(self, positive_indexes, negative_indexes):
        cls = []
        for i in range(len(positive_indexes)):
            if positive_indexes[i] == True:
                cls.append('positive')
            elif negative_indexes[i] == True:
                cls.append('negative')
            else:
                cls.append('neutral')
        return cls

    def plot_personality_3d(self, columns, c):

        personality = self.extract_columns(columns)
        print(personality.shape)
        x = np.mean(personality[:,0,:], axis=1)
        y = np.mean(personality[:,1,:], axis=1)
        z = np.mean(personality[:,2,:], axis=1)
        
        # Creating figure
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        
        print(personality[:,0].shape)

        # Creating plot
        ax.scatter(x, y, z, c=c)
        plt.title("simple 3D scatter plot")

        plt.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # show plot
        plt.show()

    def detect_change(self):
        SAH = self.data[:,self.index_sah,:]
        before_bdi = []
        after_bdi = []
        NEO_Neuroticism = []
        NEO_Extraversion = []
        NEO_Openness = [] 
        NEO_Agreeableness = [] 
        NEO_Conscientiousness = []

        for i in range(SAH.shape[0]):
            for j in range(SAH.shape[1]-1):
                if (SAH[i, j] == 1) & (SAH[i, j+1] == 0):
                    before_bdi.append(self.data[i, self.index_bdi, j])
                    after_bdi.append(self.data[i, self.index_bdi, j+1])
                    NEO_Neuroticism.append(self.data[i, self.columns.index('NEO_Neuroticism'), j])
                    NEO_Extraversion.append(self.data[i, self.columns.index('NEO_Extraversion'), j])
                    NEO_Openness.append(self.data[i, self.columns.index('NEO_Openness'), j])
                    NEO_Agreeableness.append(self.data[i, self.columns.index('NEO_Agreeableness'), j])
                    NEO_Conscientiousness.append(self.data[i, self.columns.index('NEO_Conscientiousness'), j])

        df = pd.DataFrame(columns=['before_sah_bdi','after_sah_bdi', 'NEO_Neuroticism', 'NEO_Extraversion', 'NEO_Openness', 'NEO_Agreeableness', 'NEO_Conscientiousness'], 
                          data=np.vstack([before_bdi, after_bdi, NEO_Neuroticism, NEO_Extraversion, NEO_Openness, NEO_Agreeableness, NEO_Conscientiousness]).T)

        return df
    
