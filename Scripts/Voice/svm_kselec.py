# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 02:22:00 2019

@author: Khizar Anjum

This file is meant for the application of SVM on k selections
from the mfccData with corrected indices

k selections involves the min(M1, M2, ..., MN) approach. This approach is developed 
to deal with variable number of recordings in each patients. According to this 
approach, k selections means that only k recordings (selected without replacement)
from each participant (voice) will be considered for further learning. Participants
with lower than k recordings will be dropped.
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from spline_functions import k_selections

#%% Using k_selections
mfccSeries = pd.read_csv('..\\..\\Data\\VoiceData\\splineData\\SeriesX_spline.csv')
X_train = np.load('..\\..\\Data\\VoiceData\\splineData\\X_spline_train.npy')
X_val = np.load('..\\..\\Data\\VoiceData\\splineData\\X_spline_val.npy')
X_test = np.load('..\\..\\Data\\VoiceData\\splineData\\X_spline_test.npy')

k = 3;
X_train, Y_train = k_selections(k, X_train, mfccSeries, 'train')
X_val, Y_val = k_selections(k, X_val, mfccSeries, 'val')
X_test, Y_test = k_selections(k, X_test, mfccSeries, 'test')
#%%
le = preprocessing.LabelEncoder()
label_tr = le.fit_transform(Y_train.values)
label_va = le.fit_transform(Y_val.values)
label_te = le.fit_transform(Y_test.values)


#%%
clf = SVC(gamma='auto')
clf.fit(X_train, label_tr) 

#%%
y_pred = clf.predict(X_test)


