## This file is meant to replicate the exact same process, P. Schwab has implemented
# for the voice data. 
# This file extracts trian, test and validation sets from rep_voice_cohort.csv 
# and packs them into datasets on which analysis can be done
# it consists of three sections, each of which have their own tasks
# dated 04-01-2019

#%% Importing libararies
import pandas as pd
import numpy as np 
#from pydub import AudioSegment
#import scipy.signal as sg
import sys
from dfa import dfa
from librosa.feature import mfcc

#%% Loading dataset!
v_data = pd.read_csv('..\\..\\Data\\VoiceData\\rep_voice_cohort.csv',index_col=[0,1])

#%% SECTION 1
## NOTE: IF YOU ARE WORKING ON rep20dsData, YOU CAN SAFELY SKIP NEXT TWO BLOCKS
## LOAD DATA FROM ..\\..\\Data\\VoiceData\\rep20dsData\\

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

"""
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
        update_progress((i+1)/len(v_data.loc[x,:]))
    np.save('..\\..\\Data\\VoiceData\\rep20dsData\\X_'+x+'_sec'+str(k)+'.npy',X)
"""   
#%% SECTION 2 RF FEATURE GENERATION
# this section generates a full dataset which will contain all features form Aurora et al. 
    
#%% functions required for feature extraction!
def tkeo(a):
    # author : lvanderlinden
    #function calculates Teager-Kaiser Energy operator!

	"""
	Calculates the TKEO of a given recording by using 2 samples.
	See Li et al., 2007
	Arguments:
	a 			--- 1D numpy array.
	Returns:
	1D numpy array containing the tkeo per sample
	"""

	# Create two temporary arrays of equal length, shifted 1 sample to the right
	# and left and squared:
	i = a[1:-1]*a[1:-1]
	j = a[2:]*a[:-2]

	# Calculate the difference between the two temporary arrays:
	aTkeo = i-j

	return aTkeo
"""
def find_f0(a,Fs):
    #author: Khizar Anjum
    
    """"""
    Estimates the fundamental frequency from a given raw PCM audio recording
    it uses the YIN algorithm as written in this link:
    http://recherche.ircam.fr/anasyn/roebel/amt_audiosignale/VL5.pdf
    on section 3.1
    Arguments:
    a           --- 1D numpy array.
    Fs          --- Sampling frequency of a
    Returns:
    F0          --- Estimate of fundamental frequency
    """"""
    autocorr = np.correlate(a,a,mode='full')
    autocorr = autocorr[autocorr.size//2:]
    # Rw = autocorr[0]
    # Rwd = autocorr;
"""    
    

#%% building a feature extractor from raw audio for RFs
# six types of features are extracted: 1. detrended fluctuation analysis, 2. recurrence
# period density entropy, 3. teager-kaiser energy operator, 4. jitter, 5. shimmer
# 6. mfccs
def feature_extractor(data,Fs):
    #expecting data to be a matrix of size (n_samples, raw_audio_length)
    
    for i,audio in enumerate(data):
        _,temp,_ = dfa(audio); #using default values for now!
        temp = np.hstack((temp, np.mean(tkeo(audio)))) #because mean is used in paper
        temp = np.hstack((temp, np.reshape(mfcc(y=audio,sr=Fs,n_mfcc=40),(-1,))))
        if(i == 0): y = temp
        else: y = np.vstack((y,temp))
    return y
#%%  loads rep20dsData and extracts features using feature extractor!

parts = 500;
X = [];
err_samples = [];
for x in v_data.index.levels[0]:
    for i in range(int(len(v_data.loc[x,:].index)/parts)):
        try:
            if(i==0):X = feature_extractor(np.load('..\\..\\Data\\VoiceData\\rep20dsData\\X_'+x+'_sec0.npy'),Fs=2205)
            else:X = np.vstack((X,feature_extractor(np.load('..\\..\\Data\\VoiceData\\rep20dsData\\X_'+x+'_sec'+str(i)+'.npy'),Fs=2205)))
            update_progress((i+1)/int(len(v_data.loc[x,:].index)/parts))
        except Exception as e:
            print(e);
            print('moving on!');
            err_samples.append((x,i));
    np.save('..\\..\\Data\\VoiceData\\repRFData\\X_'+x+'.npy',X)
    X = [];
    print('X_'+x+'.npy saved!')
del X
np.save('..\\..\\Data\\VoiceData\\repRFData\\err_samples.npy',err_samples)