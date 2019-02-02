# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:21:43 2019

@author: Khizar Anjum

This file is meant for data generation for further processing in spline.py. 
The data generation is necessary becuase until now, the spline filter was being
applied on mfccData_cind which is already in freq domain (LAME!)
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
mfccData = np.load('..\\..\\Data\\VoiceData\\mfccData\\X_mfcc.npy')
mfccData = np.delete(mfccData,nan_ind,axis=0)