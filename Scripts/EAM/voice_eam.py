# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:33:02 2019

@author: Khizar Anjum

This file is meant to implement an EAM model as implemented in P. Schwab's work 
but this EAM model would only aggregate or compile results from only one mode
of data i.e. voice data. 
basically, I will be using two data (repRFData or mfccData) and just couple those
RF models with a sequential RNN (BLSTM) network in order to see the difference.

"""
#%% Importing Libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from scipy import stats

#%% Importing the mfccData which is indexed correctly now!
X_train = np.load('..\\..\\Data\\VoiceData\\mfccData\\X_mfcc_train_cind.npy')
X_val = np.load('..\\..\\Data\\VoiceData\\mfccData\\X_mfcc_val_cind.npy')
X_test = np.load('..\\..\\Data\\VoiceData\\mfccData\\X_mfcc_test_cind.npy')
mfccSeries = pd.read_csv('..\\..\\Data\\VoiceData\\mfccData\\SeriesX_mfcc_cind.csv')

#%% Lets prepare the data!
Y_train = mfccSeries['professional-diagnosis'][mfccSeries['split'] == 'train']
Y_val = mfccSeries['professional-diagnosis'][mfccSeries['split'] == 'val']
Y_test = mfccSeries['professional-diagnosis'][mfccSeries['split'] == 'test']

le = preprocessing.LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_val = le.fit_transform(Y_val)
Y_test = le.fit_transform(Y_test)

X_train = X_train.reshape((X_train.shape[0],-1))
X_test = X_test.reshape((X_test.shape[0],-1))
X_val = X_val.reshape((X_val.shape[0],-1))

#%% Training the specific RF model
rfclf = RandomForestClassifier(n_estimators = 150, max_depth = 25, random_state = 42, max_features = 'log2')
rfclf.fit(X_train,Y_train)
print(rfclf.score(X_test,Y_test))
print(rfclf.score(X_train,Y_train))
#test comes out to be 0.6241933063184751
#%% Calculating next inputs and labels
X_train_eam = pd.DataFrame(index= mfccSeries['healthCode'][mfccSeries['split'] == 'train'],\
                           data = rfclf.predict(X_train))
X_val_eam = pd.DataFrame(index= mfccSeries['healthCode'][mfccSeries['split'] == 'val'],\
                         data = rfclf.predict(X_val))
X_test_eam = pd.DataFrame(index= mfccSeries['healthCode'][mfccSeries['split'] == 'test'],\
                          data = rfclf.predict(X_test))

Y_train_eam = mfccSeries[mfccSeries['split'] == 'train'].drop_duplicates(subset = 'healthCode')['professional-diagnosis']
Y_val_eam = mfccSeries[mfccSeries['split'] == 'val'].drop_duplicates(subset = 'healthCode')['professional-diagnosis']
Y_test_eam = mfccSeries[mfccSeries['split'] == 'test'].drop_duplicates(subset = 'healthCode')['professional-diagnosis']

#%% Making the model!
model = Sequential()
model.add(Bidirectional(LSTM(16),input_shape=(None,1))) #
#model.add(Bidirectional(LSTM(16)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#%% Lets train the model
for i,j in enumerate(np.unique(X_train_eam.index.values)):
    print('Training on %d sample:'%(i+1))
    if(np.ndim(X_train_eam.loc[j].values) == 2): train_in = np.squeeze(X_train_eam.loc[j].values)
    else: train_in = X_train_eam.loc[j].values
    model.fit(np.expand_dims(np.expand_dims(train_in,axis=0),axis=2), \
              np.expand_dims(np.expand_dims(np.array(int(Y_train_eam.iloc[i])),axis=0),axis=1),\
          batch_size=10,\
          epochs=10,\
          validation_data=(np.expand_dims(np.expand_dims(np.squeeze(X_val_eam.loc[X_val_eam.index[0]].values),axis=0),axis=2),\
                           np.expand_dims(np.expand_dims(np.squeeze(np.array(int(Y_val_eam.iloc[0]))),axis=0),axis=1)))
#%% Predictions from the model
y_pred = []
for i,j in enumerate(np.unique(X_test_eam.index.values)):
    if(np.ndim(X_test_eam.loc[j].values) == 2): test_in = np.squeeze(X_test_eam.loc[j].values)
    else: test_in = X_test_eam.loc[j].values
    y_pred.append(model.predict(np.expand_dims(np.expand_dims(test_in,axis=0),axis=2)))
    
#%% lets check the predictions!
y_pred = np.round(np.ravel(y_pred))    
Y_test_eam = le.fit_transform(Y_test_eam.values)
print(((Y_test_eam - y_pred)**2).sum()/len(y_pred))
#comes out to be 0.3140495867768595
#%% The mode predictions
y_pred = []
for i,j in enumerate(np.unique(X_test_eam.index.values)):
    temp,_ = stats.mode(np.squeeze(X_test_eam.loc[j].values))
    y_pred.append(temp)
y_pred = np.round(np.ravel(y_pred))   
print(((Y_test_eam - y_pred)**2).sum()/len(y_pred))
#comes out to be ~0.31074380165289256
#%% SECTION 2:
"""
THIS IS THE SECOND SECTION OF THIS PYTHON FILE AND IT DEALS WITH THE APPLICATION
OF THE LSTM EAM ON THE repRFData. 
IT DOES NOT SHARE ANYTHING WITH ALL THE ABOVE SECTIONS WITH THE EXCEPTION OF IMPORTING
LIBRARIES
"""
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
print(rfclf.score(X_test,Y_test))
# comes out to be 0.5847076461769115
#%%
X_train_eam = pd.DataFrame(index= series_data.loc['train',:]['healthCode'],\
                           data = rfclf.predict(X_train))
X_val_eam = pd.DataFrame(index= series_data.loc['val',:]['healthCode'],\
                         data = rfclf.predict(X_val))
X_test_eam = pd.DataFrame(index= series_data.loc['test',:]['healthCode'],\
                          data = rfclf.predict(X_test))

Y_train_eam = series_data.loc['train',:].drop_duplicates(subset = 'healthCode')['professional-diagnosis']
Y_val_eam = series_data.loc['val',:].drop_duplicates(subset = 'healthCode')['professional-diagnosis']
Y_test_eam = series_data.loc['test',:].drop_duplicates(subset = 'healthCode')['professional-diagnosis']

#%% Making the model!
model = Sequential()
model.add(Bidirectional(LSTM(16),input_shape=(None,1))) #
#model.add(Bidirectional(LSTM(16)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#%% Lets train the model
for i,j in enumerate(np.unique(X_train_eam.index.values)):
    print('Training on %d sample:'%(i+1))
    if(np.ndim(X_train_eam.loc[j].values) == 2): train_in = np.squeeze(X_train_eam.loc[j].values)
    else: train_in = X_train_eam.loc[j].values
    model.fit(np.expand_dims(np.expand_dims(train_in,axis=0),axis=2), \
              np.expand_dims(np.expand_dims(np.array(int(Y_train_eam.iloc[i])),axis=0),axis=1),\
          batch_size=10,\
          epochs=10,\
          validation_data=(np.expand_dims(np.expand_dims(np.squeeze(X_val_eam.loc[X_val_eam.index[0]].values),axis=0),axis=2),\
                           np.expand_dims(np.expand_dims(np.squeeze(np.array(int(Y_val_eam.iloc[0]))),axis=0),axis=1)))
#%% Predictions from the model
y_pred = []
for i,j in enumerate(np.unique(X_test_eam.index.values)):
    if(np.ndim(X_test_eam.loc[j].values) == 2): test_in = np.squeeze(X_test_eam.loc[j].values)
    else: test_in = X_test_eam.loc[j].values
    y_pred.append(model.predict(np.expand_dims(np.expand_dims(test_in,axis=0),axis=2)))
#%% lets check the predictions!
y_pred = np.round(np.ravel(y_pred))    
Y_test_eam = le.fit_transform(Y_test_eam.values)
print(((Y_test_eam - y_pred)**2).sum()/len(y_pred))
#comes out to be about 0.5098814229249012
#%%
y_pred = []
for i,j in enumerate(np.unique(X_test_eam.index.values)):
    temp,_ = stats.mode(np.squeeze(X_test_eam.loc[j].values))
    y_pred.append(temp)
y_pred = np.round(np.ravel(y_pred))    
print(((Y_test_eam - y_pred)**2).sum()/len(y_pred))
#comes out to be 0.48221343873517786