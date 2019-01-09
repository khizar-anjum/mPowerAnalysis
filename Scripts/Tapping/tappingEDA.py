## This file is meant to replicate the exact same process, P. Schwab has implemented
# for the tapping data. The goal is to extract the same train, test and validation
# sets!
# dated 08-01-2019

#%% importing libraries
import pandas as pd
import numpy as np

#%%
# lets load the demographics file here
demo_data = pd.read_csv('..\\..\\Data\\Demographics\\rep_study_cohort.csv',index_col=0)

#lets load the tapping data file
tapping_data = pd.read_csv('..\\..\\Data\\TappingData\\Tapping Activity.csv')

#%%
#some cleaning underway
tapping_data = tapping_data.dropna(subset=['medTimepoint'])
tapping_data.drop(columns = ['ROW_ID','ROW_VERSION','appVersion','phoneInfo','createdOn'],inplace=True)
#removing patients who performed test just after medication
tapping_data = tapping_data[tapping_data['medTimepoint'] != 'Just after Parkinson medication (at your best)']
#removing patients who performed test another time
tapping_data = tapping_data[tapping_data['medTimepoint'] != 'Another time']
#now, only these are the audio files that are necessary!
#%%
#lets take intersection. i.e. the patients who also performed the tapping test
c1 = tapping_data['healthCode'].unique().tolist()
c2 = demo_data['healthCode'].unique().tolist()
c3 = list(filter(lambda x: x in c1, c2))
print(len(c2))
#%%
#now, lets sort the tapping data patients using some multiIndex magic
tapping_data = tapping_data.set_index(['healthCode','recordId']).sort_index()
#tapping_data.head()
tapping_data = tapping_data.loc[c3,:]
tapping_data.head()
#%%
#constructing merged series with respect to healthCode
merged_data = demo_data.set_index('healthCode').loc[c3]
#len(tapping_data)
#so, so far Y contains the target variables with respoect to healthCode
#and tapping_data contains all the healthCode audio_file codes and recordIds (which
#contain the repitition of activity by the same person)

#%%
#preprocessing for partitioning 1262 merged participants into groups of 70-20-10 
#on the basis 
from sklearn.model_selection import StratifiedShuffleSplit
sk1 = StratifiedShuffleSplit(n_splits = 1, test_size=0.20)
sk2 = StratifiedShuffleSplit(n_splits = 1, test_size=0.125)
y = merged_data['age']

#just a check for eliminating an age group with only one member
minclass = np.argmin([(merged_data['age']==x).sum() for x in np.unique(y)])
X = merged_data.index
X = X[(merged_data['age'] != np.unique(y)[minclass]).values]
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
#this block is to now partition the merged dataframe according to the values found
#resetting index stuff
tapping_data.index = tapping_data.index.droplevel(1)
mylist = ['age','gender','professional-diagnosis'];
tapping_data[mylist] = merged_data[mylist]
#generating data!
d_test = tapping_data.loc[y_test.index.values]
d_train = tapping_data.loc[y_train.index.values]
d_val = tapping_data.loc[y_val.index.values]
#%%    
#saving my all the tapping data cohort into csv
frames = [d_train, d_val, d_test];
rep_tapping_cohort = pd.concat(frames, keys = ['train','val','test'])
rep_tapping_cohort.to_csv('..\\..\\Data\\TappingData\\rep_tapping_cohort.csv')
#this csv contains train, test and validation sets for the tapping tests!