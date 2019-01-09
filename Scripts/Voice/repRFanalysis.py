# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:59:36 2019

@author: Khizar Anjum
"""
#this file is meant for RF application on repRFData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve, auc

#%% Importing our datasets!
X_val = np.load('..\\..\\Data\\VoiceData\\repRFData\\X_val.npy')
X_train = np.load('..\\..\\Data\\VoiceData\\repRFData\\X_train.npy')
X_test = np.load('..\\..\\Data\\VoiceData\\repRFData\\X_test.npy')

#%% Importing labels
series_data = pd.read_csv('..\\..\\Data\\VoiceData\\repRFData\\SeriesrepRFData.csv',index_col = [0,1])
Y_train = series_data.loc['train',:]['professional-diagnosis'].values
Y_val = series_data.loc['val',:]['professional-diagnosis'].values
Y_test = series_data.loc['test',:]['professional-diagnosis'].values
#%%
le = preprocessing.LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_val = le.fit_transform(Y_val)
Y_test = le.fit_transform(Y_test)

#%%
rfclf = RandomForestClassifier(n_estimators = 1024, max_depth = 5, random_state = 42, max_features = 'log2')
rfclf.fit(X_train,Y_train)
y_pred = rfclf.predict(X_test);

#%%
y_prob = rfclf.predict_proba(X_test);
print('AUC ROC:',roc_auc_score(Y_test,y_prob[:,1]))
print('AUC PR:',average_precision_score(Y_test,y_prob[:,1]))
print('F1:',f1_score(Y_test,y_pred))
print('Accuracy:',accuracy_score(Y_test,y_pred))
print(rfclf.score(X_test,Y_test))

#%%
fpr, tpr, thresholds = roc_curve(Y_test,y_prob[:,1])
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC (AUC = %0.2f)'%roc_auc)