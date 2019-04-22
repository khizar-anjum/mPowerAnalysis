# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 04:17:21 2019

@author: m.khizer
"""

#%% Importing Libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score

#%%
def k_selections(k, data, series, split):
    """
    This function takes in data and returns data with k selections procedure 
    done on it as given in the intro of this file
    
    Parameters:
    -----------
        k : an integer
            The number of recordings to keep for each patient  
        data : a numpy matrix
            A matrix of shape (n_recordings, (whatever the shape)) but of 
            the split given in  
        series : a pandas dataframe
            A dataframe containing info about data  
        split : a simple string
            The split which is being input to the function  
    
    Returns:
    --------
        X : A numpy matrix
            data matrix after the procedure of size (n_participants*k, (shape))  
        Y : A pandas series containing target values for recordings
            Target values about data  
    """
    series = series[series['split'] == split].reset_index(drop=True)
    reps = series['healthCode'].value_counts()
    reps = reps[reps >= k]
    series = series.reset_index().set_index('healthCode')
    Y = pd.DataFrame(columns = series.columns)
    for i,_ in enumerate(reps):
        Y = Y.append(series.loc[reps.index[i]].iloc[np.random.randint(0,reps[i],size=(k,))])
    X = data[Y['index'].values.tolist()]
    Y = Y['professional-diagnosis']
    return X,Y

#%% Importing the mfccData which is indexed correctly now!
X_train = np.load('..\\..\\Data\\VoiceData\\mfccData\\X_mfcc_train_cind.npy')
X_val = np.load('..\\..\\Data\\VoiceData\\mfccData\\X_mfcc_val_cind.npy')
X_test = np.load('..\\..\\Data\\VoiceData\\mfccData\\X_mfcc_test_cind.npy')
mfccSeries = pd.read_csv('..\\..\\Data\\VoiceData\\mfccData\\SeriesX_mfcc_cind.csv')

k = 5;
X_train, Y_train = k_selections(k, X_train, mfccSeries, 'train')
X_val, Y_val = k_selections(k, X_val, mfccSeries, 'val')
X_test, Y_test = k_selections(k, X_test, mfccSeries, 'test')
#%%
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
print(roc_auc_score(Y_test,rfclf.predict_proba(X_test)[:,1]))
print(accuracy_score(Y_test, rfclf.predict(X_test)))

#%%
label_ = Y_test; y_ = rfclf.predict_proba(X_test)[:,1]
true_positives = np.sum(np.round(np.clip(label_ * y_, 0, 1)))
possible_positives = np.sum(np.round(np.clip(label_, 0, 1)))
true_negatives = np.sum(np.round(np.clip((1-label_) * (1-y_), 0, 1)))
possible_negatives = np.sum(np.round(np.clip(1-label_, 0, 1)))
spec =  true_negatives / (possible_negatives + 1e-10)
sens =  true_positives / (possible_positives + 1e-10)
print(sens)
print(spec)
