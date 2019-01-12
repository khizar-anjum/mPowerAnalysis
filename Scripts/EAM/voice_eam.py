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
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

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
model.add(Bidirectional(LSTM(16,return_sequences=True)))
model.add(Bidirectional(LSTM(16)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#%% Lets train the model
for i,j in enumerate(np.unique(X_train_eam.index.values)):
    print('Training on %d sample:'%(i+1))
    model.fit(np.squeeze(X_train_eam.loc[j].values), int(Y_train_eam.iloc[i]),\
          batch_size=10,\
          epochs=10,\
          validation_data=[np.squeeze(X_val_eam.loc[X_val_eam.index[0]].values), int(Y_val_eam.iloc[0])])
