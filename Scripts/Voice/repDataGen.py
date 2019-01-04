## This file is meant to replicate the exact same process, P. Schwab has implemented
# for the voice data. 
# This file extracts trian, test and validation sets from rep_voice_cohort.csv 
# and packs them into a dataset on which analysis can be done
# this dataset means a full dataset which will contain all features form Aurora et al.
# dated 04-01-2019

#%% Importing libararies
import pandas as pd
import numpy as np 
from pydub import AudioSegment
import scipy.signal as sg
import sys

#%% Loading dataset!
v_data = pd.read_csv('..\\..\\Data\\VoiceData\\rep_voice_cohort.csv',index_col=[0,1])

## NOTE: IF YOU ARE WORKING ON rep20dsData, YOU CAN SAFELY SKIP NEXT TWO BLOCKS
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
#%% this block converts audio files into raw_audio format and decimates them
pad = lambda a,i : a[0:i] if len(a) > i else np.concatenate((a, [0] * (i-len(a)))) #for fixing things to a fixed length
sample_len = 22050; #sample Fs is 44100. After 20x decimation, a 10-sec recording should have 44.1k Samples
err_samples = [];
parts = 500;
#X = np.array((sample_len));
#lets start extracting samples!
for x in v_data.index.levels[0]:
    j = True;
    k = 0;
    for i,_ in enumerate(v_data.loc[x,:].index):
        path = '..\\..\\Data\\VoiceData\\Audio\\'+str(v_data.loc[x,:]['audio_audio.m4a'].iloc[i]) + '.m4a';
        try:
            audio = np.array(AudioSegment.from_file(path).get_array_of_samples().tolist());
            audio = audio/audio.max() #Normalize the PCM waveform
            deci = sg.decimate(audio,q=20) #lets decimate by 20 
            deci = pad(deci,sample_len); #lets fix the sample length! so that we have no problems in concatenation!
            #plt.plot(np.abs(np.fft.fftshift(np.fft.fft(deci)))) #If you want to visualize the fft of the thing!
            if(j is True): 
                X = deci;
                X = np.expand_dims(X,axis=0)
                j = False;
            else: X = np.vstack((X,deci))
            
            if(X.shape[0] % parts == 0):
                np.save('..\\..\\Data\\VoiceData\\rep20dsData\\X_'+x+'_sec'+str(k)+'.npy',X)
                k = k + 1;
                j = True;
        except:
            err_samples.append((x,i));
        update_progress((i+1)/len(v_data))
    np.save('..\\..\\Data\\VoiceData\\rep20dsData\\X_'+x+'_sec'+str(k)+'.npy',X)
    
#%%
    