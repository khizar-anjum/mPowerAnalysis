# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:21:43 2019

@author: Khizar Anjum

This file is meant for data generation for further processing in spline.py. 
The data generation is necessary becuase until now, the spline filter was being
applied on AData_cind which is already in freq domain (LAME!)
hence, now I am working towards doing stuff on this data!

I am going to extract the incoming data from the AnalysisData and probably do
some other processing before cleanly dividing it into stuff!

"""

#Importing Libraries
import pandas as pd
import numpy as np


#%% Loading analysis Data
ASeries = pd.read_csv('..\\..\\Data\\VoiceData\\AnalysisData\\SeriesAnalysisData.csv')
for i in range(71):
    if(i == 0):
        Adata = np.load('..\\..\\Data\\VoiceData\\AnalysisData\\X_sec'+str(i)+'.npy')[:,::2]
    else: 
        Adata = np.vstack((Adata,np.load('..\\..\\Data\\VoiceData\\AnalysisData\\X_sec'\
                                        +str(i)+'.npy')[:,::2]))

#%% removing nan from age
nan_ind = ASeries['age'].index[ASeries['age'].apply(np.isnan)]
ASeries = ASeries.dropna(subset = ['age'])
Adata = np.delete(Adata,nan_ind,axis=0)

#%% reducing data to individual patients
patData = ASeries.drop_duplicates(subset = ['healthCode']).set_index('healthCode')
patData.head()
#%%
#preprocessing for partitioning 3024 participants into groups of 70-20-10 
#on the basis of their age
from sklearn.model_selection import StratifiedShuffleSplit
sk1 = StratifiedShuffleSplit(n_splits = 1, test_size=0.20)
sk2 = StratifiedShuffleSplit(n_splits = 1, test_size=0.125)
y = patData['age']

#just a check for eliminating an age group with only one member
minclass = np.argmin([(patData['age']==x).sum() for x in np.unique(y)])
X = patData.index.values
X = X[(patData['age'] != np.unique(y)[minclass]).values]
y_ign = y[y == np.unique(y)[minclass]]
y = y[y != np.unique(y)[minclass]]

for train_index, test_index in sk1.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_temp, X_test = X[train_index], X[test_index]
    y_temp, y_test = y[train_index], y[test_index]

for train_index, test_index in sk2.split(X_temp, y_temp):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_val = X_temp[train_index], X_temp[test_index]
    y_train, y_val = y_temp[train_index], y_temp[test_index]
    
#lets add ignored entry to validation set
y_val = y_val.append(y_ign)

#%%
ASeries = ASeries.set_index('healthCode')
y = pd.concat([y_train, y_val, y_test],keys = ['train', 'val', 'test'])
y = y.to_frame().reset_index(level=0)
ASeries['split'] = y['level_0']
ASeries.head()
#%%
X_train = Adata[(ASeries.split == 'train').values,:]
X_val = Adata[(ASeries.split == 'val').values,:]
X_test = Adata[(ASeries.split == 'test').values,:]
#%%
np.save('..\\..\\Data\\VoiceData\\splineData\\X_spline_train.npy',X_train)
np.save('..\\..\\Data\\VoiceData\\splineData\\X_spline_test.npy',X_test)
np.save('..\\..\\Data\\VoiceData\\splineData\\X_spline_val.npy',X_val)
#%%
ASeries.to_csv('..\\..\\Data\\VoiceData\\splineData\\SeriesX_spline.csv')
