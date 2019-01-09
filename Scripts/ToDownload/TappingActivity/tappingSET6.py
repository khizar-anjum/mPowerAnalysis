# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:59:17 2019

@author: Khizar Anjum
"""
#the aim of this file is to let me download the Tapping Activity Data 
#from synapse. 
"""
#%% Getting to know the data we're downloading
import json
with open('..\\..\\..\\Data\\TappingData\\Accel\\5395782.json','r') as f:
    info1 = json.load(f)

"""
#%% Importing Libraries!
import synapseclient
import shutil

#%% Logging in
syn = synapseclient.Synapse()
syn.login()

#%%
tapping_table = 'syn5511439';
columns = ['accel_tapping.json.items','tapping_results.json.TappingSamples'];
folder = ['Accel','Tapping'];
for i,_ in enumerate(columns):
    results = syn.tableQuery('SELECT * FROM '+tapping_table+' LIMIT 13887 OFFSET 65000')
    file_map = syn.downloadTableColumns(results, columns[i])
    for file_handle_id, path in file_map.items():
        shutil.move(path,'D:\\Projects\\Parkinson Diagnosis\\mPowerAnalysis\\Data\\TappingData\\' + folder[i] + '\\'+\
                str(file_handle_id)+'.json')
syn.logout()