# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 01:09:28 2019

@author: Khizar Anjum

this file contains functions that are meant to be used in spline.py file
"""
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import backend as K

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


#%% Useful functions
#defining useful functions
def th_linspace(start,end,n):
        n = int(n)
        a = np.arange(float(n),dtype=np.float32)/(float(n))
        a*=(end-start)
        return a+start

#defining useful functions
class tf_hermite_complex:
    def __init__(self,S,deterministic,initialization,renormalization,chirplet=0):
        """S: integer (the number of regions a.k.a piecewise polynomials)
        deterministic: bool (True if hyper-parameters are learnable)
        initialization: 'random','gabor','random_apodized' 
        renormalization: fn(x):norm(x) (theano function for filter renormalization)"""
        self.S               = S
        self.chirplet        = chirplet
        self.renormalization = renormalization
        self.deterministic   = deterministic
#        T                    = K.constant([],dtype='int32')
        self.mask            = np.ones(S,dtype='float32') # MASK will be used to apply boundary conditions
        self.mask[[0,-1]]    = 0 # boundary conditions correspond to 0 values on the boundaries
        if(initialization=='gabor'):
            aa          = np.ones(S)
            aa[::2]    -= 2
            thetas_real = aa*np.hanning(S)**2
            thetas_imag = np.roll(aa,1)*np.hanning(S)**2
            gammas_real = np.zeros(S)
            gammas_imag = np.zeros(S)
            c           = np.zeros(1)
        elif(initialization=='random'):
            thetas_real = np.random.rand(S)*2-1
            thetas_imag = np.random.rand(S)*2-1
            gammas_real = np.random.rand(S)*2-1
            gammas_imag = np.random.rand(S)*2-1
            if(chirplet):
                c   = np.zeros(1)
            else:
                c   = np.zeros(1)
        elif(initialization=='random_apodized'):
            thetas_real = (np.random.rand(S)*2-1)*np.hanning(S)**2
            thetas_imag = (np.random.rand(S)*2-1)*np.hanning(S)**2
            gammas_real = (np.random.rand(S)*2-1)*np.hanning(S)**2
            gammas_imag = (np.random.rand(S)*2-1)*np.hanning(S)**2
            if(chirplet):
                    c   = np.zeros(1)
            else:
                    c   = np.zeros(1)
        else:
            sys.exit('ERROR! Please input correct name of initialization required!')
                    
        if(chirplet):
            self.c            = c
            self.thetas_real  = thetas_real
            self.thetas_imag  = thetas_imag
            self.gammas_real  = gammas_real
            self.gammas_imag  = gammas_imag
        
        # NOW CREATE THE POST PROCESSED VARIABLES
        self.ti           = np.expand_dims(th_linspace(0,1,self.S),axis=1)# THIS REPRESENTS THE MESH
        if(self.chirplet):
            thetas_real = self.thetas_real
            thetas_imag = self.thetas_imag
            gammas_real = self.gammas_real
            gammas_imag = self.gammas_imag
            c = self.c
            TT          = c.repeat(S)*float(2*3.14159)*(th_linspace(0,1,S)**2)
            TTc         = c.repeat(S)*float(2*3.14159)*th_linspace(0,1,S)
            thetas_real = thetas_real*np.cos(TT)-thetas_imag*np.sin(TT)
            thetas_imag = thetas_imag*np.cos(TT)+thetas_real*np.sin(TT)
            gammas_real = gammas_real*np.cos(TT)-gammas_imag*np.sin(TT)-thetas_real*np.sin(TT)*TTc-thetas_imag*np.cos(TT)*TTc
            gammas_imag = gammas_imag*np.cos(TT)+gammas_real*np.sin(TT)+thetas_real*np.cos(TT)*TTc-thetas_imag*np.sin(TT)*TTc
            
        else:
            thetas_real = self.thetas_real
            thetas_imag = self.thetas_imag
            gammas_real = self.gammas_real
            gammas_imag  = self.gammas_imag
        #NOW APPLY BOUNDARY CONDITION
        
        self.pthetas_real   = np.expand_dims(((thetas_real-thetas_real[1:-1].mean())*self.mask),axis=1)
        self.pthetas_imag   = np.expand_dims(((thetas_imag-thetas_imag[1:-1].mean())*self.mask),axis=1)
        self.pgammas_real   = np.expand_dims((gammas_real*self.mask),axis=1)
        self.pgammas_imag   = np.expand_dims((gammas_imag*self.mask),axis=1)
    def get_filters(self,T):
            #"""method to obtain one filter with length T"""
        t=np.expand_dims(th_linspace(0,1,T),axis=0)#THIS REPRESENTS THE CONTINOUS TIME (sampled)
        #COMPUTE FILTERS BASED ONACCUMULATION OF WINDOWED INTERPOLATION
        real_filter     = self.interp((t-self.ti[:-1])/(self.ti[1:]-self.ti[:-1]),
            self.pthetas_real[:-1],self.pgammas_real[:-1],self.pthetas_real[1:],self.pgammas_real[1:]).sum(0)
        imag_filter     = self.interp((t-self.ti[:-1])/(self.ti[1:]-self.ti[:-1]),
            self.pthetas_imag[:-1],self.pgammas_imag[:-1],self.pthetas_imag[1:],self.pgammas_imag[1:]).sum(0)
        # RENORMALIZE
        filt = self.renormalization(real_filter)+self.renormalization(imag_filter)
        real_filter     = real_filter/(filt+0.00001)
        imag_filter     = imag_filter/(filt+0.00001)
        return real_filter,imag_filter
    def interp(self,t,pi,mi,pip,mip):
        values = ((2*t**3-3*t**2+1)*pi+(t**3-2*t**2+t)*mi+(-2*t**3+3*t**2)*pip+(t**3-t**2)*mip)
        mask   = np.greater_equal(t,0).astype(float)*np.less(t,1).astype(float)
        return values*mask

def create_center_filter_complex(filter_length,max_length,class_function):
    deltaT = max_length-filter_length
    #print (deltaT)
    real_f,imag_f =class_function(filter_length)
    filters = np.concatenate([real_f.reshape((1,-1)),imag_f.reshape((1,-1))],axis=0)
    #	if(deltaT!=0):
    return np.roll(np.concatenate([filters,np.zeros((2,deltaT),dtype='float32')],axis=1),int(deltaT/2),axis=1)

#important functions
def create_filter_banks_complex(filter_class,N,J,Q):
    scales = np.array(2**(np.arange(J*Q+1,dtype='float32')/Q)).astype('float32')#all the scales
    Ts     = np.array([N*scale for scale in scales]).astype('int32')#all the filter size support
    #	Ts_t   = theano.shared(Ts)
    #print ("Lengths of the filters:",Ts)
    #	filters = [filter_class.get_filters(t) for t in Ts]
    filter_bank = np.concatenate([create_center_filter_complex(i,Ts[-1],filter_class.get_filters) for i in Ts[:-1]],axis=0)
    #        filter_bank,_=theano.scan(fn = lambda filter_length,max_length: create_center_filter_complex(filter_length,max_length,filter_class.get_filters),sequences=Ts[:-1],non_sequences=Ts[-1])
    #	filter_bank=filter_bank[-1]
    return filter_bank[::2,:],filter_bank[1::2,:],Ts[-1]#filter_bank[::2],filter_bank[1::2],Ts[-1]

"""
deterministic:bool, wether to update or not the hyper parameters for the splines
normalization: theano function to renormalize each of the filters individually
initialization:'gabor','random_random_apodized', initialization of the spline hyper parameters
"""
#%%
#these functions do the initialization for spline wavelets
def real_sp_initializer(shape,dtype=tf.float32):
    N,J,Q,S    = 50,2,64,190
    myfil = tf_hermite_complex(S=S,deterministic=0,renormalization=np.linalg.norm,initialization='gabor',chirplet=1);
    [real_bank,_,_] = create_filter_banks_complex(filter_class=myfil,N=N,J=J,Q=Q);
    #real_bank = np.expand_dims(real_bank.T,axis=1)
    real_bank = np.repeat(np.expand_dims(real_bank.T,axis=1),shape[1],axis=1)
    return tf.convert_to_tensor(real_bank, dtype=dtype)

def imag_sp_initializer(shape,dtype=tf.float32):
    N,J,Q,S    = 50,2,64,190
    myfil = tf_hermite_complex(S=S,deterministic=0,renormalization=np.linalg.norm,initialization='gabor',chirplet=1);
    [_,imag_bank,_] = create_filter_banks_complex(filter_class=myfil,N=N,J=J,Q=Q);
    imag_bank = np.expand_dims(imag_bank.T,axis=1)
    #imag_bank = np.repeat(imag_bank,shape[1],axis=1).reshape(T,shape[0],J*Q)
    return tf.convert_to_tensor(imag_bank, dtype=dtype)

def sensitivity(y_true, y_pred):
    y_true = y_true[:,1]; y_pred = y_pred[:,1]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    y_true = y_true[:,1]; y_pred = y_pred[:,1]
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())