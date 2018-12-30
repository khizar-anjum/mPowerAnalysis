# -*- coding: utf-8 -*-
"""
Created on Sat Nov 03 11:50:21 2018

@author: Khizar Anjum
"""

import synapseclient
import numpy as np
import sys

syn = synapseclient.login(username, password)
import shutil
#%%
def update_progress(progress):
    barLength = 30 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
#%% This segment downloads the voice activity data!
row_ids = np.arange(72137,137159);
voice_table = 'syn5511444';
columns = ['audio_audio.m4a','audio_countdown.m4a'];
folder = ['\\Audio','\\Countdown'];

for i in range(2):
    for j in row_ids:
        filepath = syn.downloadTableFile(voice_table, columns[i], downloadLocation=\
                             'E:\LUMS\Fall 2018\mPower\Data\Voice Data' + folder[i],\
                             rowId=j, versionNumber=1, \
                             ifcollision='keep.both');
        update_progress(((j+1)*(i+1))/(2*len(row_ids)));

#%% FASTER DOWNLOAD METHOD!
results = syn.tableQuery('SELECT * FROM syn5511444 LIMIT 100 OFFSET 100')
file_map = syn.downloadTableColumns(results, columns)
for file_handle_id, path in file_map.items():
    shutil.move(path,'E:\\LUMS\\Fall 2018\\mPower\\Data\\Voice Data' + folder[0] + '\\'+\
            str(file_handle_id)+'-'+path.split('\\')[-1][:-4]+'.m4a')
#%%
    '''
for file_handle_id, path in file_map.items():
    shutil.move(path,'E:\\LUMS\\Fall 2018\\mPower\\Data\\Voice Data' + folder[0] + '\\'+\
            str(file_handle_id)+'-'+path.split('\\')[-1][:-4]+'.m4a')
    '''