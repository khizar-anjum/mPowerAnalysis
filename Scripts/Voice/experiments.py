# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:15:15 2019

@author: Khizar Anjum

This file is intended for performing random experiments on the voice data in 
the hopes of finding new kinds of patterns or stuff
"""

#%% Loading Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
#%% 
"""
Experiment 1: the min(M1, M2, ..., MN) experiment!
# Here M1, M2, ..., MN represent the number of recordings by each N patients!
# The most sense for this experiment is to be performed on repRFData and mfccData_cind
"""
#%% Importing the mfccData which is indexed correctly now!
mfccSeries = pd.read_csv('..\\..\\Data\\VoiceData\\mfccData\\SeriesX_mfcc_cind.csv')
reps = mfccSeries['healthCode'].value_counts()

uniques = np.unique(reps.values)
count_reps = [(reps.values == i).sum()/len(reps) for i in uniques]
# Mode number of recordings is 3

np.average(reps.values)
# Average number of recordings is 11.57

#%% Importing the Series for repRFData
mfccSeries = pd.read_csv('..\\..\\Data\\VoiceData\\repRFData\\SeriesrepRFData.csv',index_col = [0,1])
reps = mfccSeries['healthCode'].value_counts()

uniques = np.unique(reps.values)
count_reps = [(reps.values == i).sum()/len(reps) for i in uniques]
# Mode number of recordings is 3

np.average(reps.values)
# Average number of recordings is 16.25

#%%
"""
Experiment 2: Group AnotherTime patients together
Here I intend to investigate into the increased accuracy that results when I include
patients with medTimepoint of 'Another Time'. 
"""
#%% Doing the experiment on mfccData which is correctly indexed now!
X_train = np.load('..\\..\\Data\\VoiceData\\mfccData\\X_mfcc_train_cind.npy')
X_val = np.load('..\\..\\Data\\VoiceData\\mfccData\\X_mfcc_val_cind.npy')
X_test = np.load('..\\..\\Data\\VoiceData\\mfccData\\X_mfcc_test_cind.npy')
mfccSeries = pd.read_csv('..\\..\\Data\\VoiceData\\mfccData\\SeriesX_mfcc_cind.csv')
#%% Extracting persons with only another time medtimepoints 
Y_train = mfccSeries[['professional-diagnosis','medTimepoint']][mfccSeries['split'] == 'train']
Y_val = mfccSeries[['professional-diagnosis','medTimepoint']][mfccSeries['split'] == 'val']
Y_test = mfccSeries[['professional-diagnosis','medTimepoint']][mfccSeries['split'] == 'test']

X_train = X_train[Y_train['medTimepoint'] == 'Another time']
X_val = X_val[Y_val['medTimepoint'] == 'Another time']
X_test = X_test[Y_test['medTimepoint'] == 'Another time']

Y_train = Y_train['professional-diagnosis'][Y_train['medTimepoint'] == 'Another time']
Y_test = Y_test['professional-diagnosis'][Y_test['medTimepoint'] == 'Another time']
Y_val = Y_val['professional-diagnosis'][Y_val['medTimepoint'] == 'Another time']
#%% Preparing the data!
le = preprocessing.LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_val = le.fit_transform(Y_val)
Y_test = le.fit_transform(Y_test)

X_train = X_train.reshape((X_train.shape[0],-1))
X_test = X_test.reshape((X_test.shape[0],-1))
X_val = X_val.reshape((X_val.shape[0],-1))
#%%
rfclf = RandomForestClassifier(n_estimators = 150, max_depth = 25, random_state = 42, max_features = 'log2')
rfclf.fit(X_train,Y_train)
print(rfclf.score(X_test,Y_test))
print(rfclf.score(X_train,Y_train))