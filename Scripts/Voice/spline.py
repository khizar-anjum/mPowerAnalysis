# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:58:19 2019

@author: Khizar Anjum

This file is meant for the application of Spline Filter (DNN) learning on k selections
from the mfccData with corrected indices
k selections involves the min(M1, M2, ..., MN) approach. This approach is developed 
to deal with variable number of recordings in each patients. According to this 
approach, k selections means that only k recordings (selected without replacement)
from each participant (voice) will be considered for further learning. Participants
with lower than k recordings will be dropped.

Now, the k can be different and its values can be determined emperically by trying
different values. By initial assessments in experiments.py, we see that almost 60%
of the participants have atmost 3 recordings where ~30% of the participants have 
exactly 3 recordings per participant. More information about this can be found
in evernote. 
"""
#The methods in this file are taken from "Past Project History/SplineKerasModel/
#splinefilter1d.ipynb". 

#%% Importing libraries

import sys
import pandas as pd
import numpy as np
#import keras
#from keras import backend as K
from keras.utils import to_categorical
from sklearn import preprocessing
#from keras.engine.topology import Layer
import tensorflow as tf
import matplotlib.pyplot as plt
#import pickle
#import scipy.io
#from sklearn.model_selection import train_test_split
#from keras import utils as np_utils
#from sklearn.model_selection import StratifiedKFold
from keras.callbacks import CSVLogger, EarlyStopping
#from keras.initializers import Initializer
from keras.layers import Input, Dense, Add, MaxPooling1D, Dropout, Flatten
#from keras.lyaers import Reshape
from keras.models import Model
from keras.layers.convolutional import Conv1D
from scipy.stats import mode
from sklearn.utils import class_weight

#%% WRITING IMP FUNCTIONS
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


#%% Using k_selections
mfccSeries = pd.read_csv('..\\..\\Data\\VoiceData\\splineData\\SeriesX_spline.csv')
X_train = np.load('..\\..\\Data\\VoiceData\\splineData\\X_spline_train.npy')
X_val = np.load('..\\..\\Data\\VoiceData\\splineData\\X_spline_val.npy')
X_test = np.load('..\\..\\Data\\VoiceData\\splineData\\X_spline_test.npy')

k = 3;
X_train, Y_train = k_selections(k, X_train, mfccSeries, 'train')
X_val, Y_val = k_selections(k, X_val, mfccSeries, 'val')
X_test, Y_test = k_selections(k, X_test, mfccSeries, 'test')
#%%
le = preprocessing.LabelEncoder()
label_tr = to_categorical(le.fit_transform(Y_train.values),num_classes=2)
label_va = to_categorical(le.fit_transform(Y_val.values),num_classes=2)
label_te = to_categorical(le.fit_transform(Y_test.values),num_classes=2)

X_train = np.expand_dims(X_train,axis=2)
X_test = np.expand_dims(X_test,axis=2)
X_val = np.expand_dims(X_val,axis=2)
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
    N,J,Q,S    = 50,2,256,96
    myfil = tf_hermite_complex(S=S,deterministic=0,renormalization=np.linalg.norm,initialization='gabor',chirplet=1);
    [real_bank,_,_] = create_filter_banks_complex(filter_class=myfil,N=N,J=J,Q=Q);
    real_bank = np.expand_dims(real_bank.T,axis=1)
    #real_bank = np.repeat(real_bank,shape[1],axis=1).reshape(T,shape[0],J*Q)
    return tf.convert_to_tensor(real_bank, dtype=dtype)

def imag_sp_initializer(shape,dtype=tf.float32):
    N,J,Q,S    = 50,2,256,96
    myfil = tf_hermite_complex(S=S,deterministic=0,renormalization=np.linalg.norm,initialization='gabor',chirplet=1);
    [_,imag_bank,_] = create_filter_banks_complex(filter_class=myfil,N=N,J=J,Q=Q);
    imag_bank = np.expand_dims(imag_bank.T,axis=1)
    #imag_bank = np.repeat(imag_bank,shape[1],axis=1).reshape(T,shape[0],J*Q)
    return tf.convert_to_tensor(imag_bank, dtype=dtype)
#%%
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects    

def square_activation(x):
    return K.square(x)

get_custom_objects().update({'square_activation': Activation(square_activation)})

#%% Making model in here!
def spline_model(J = 2, Q = 256, T = 200):
    inputs = Input(shape=(22050,1))
    #
    #
    x = Conv1D(filters=int(J*Q),kernel_size=int(T),padding='valid',strides=10, activation='square_activation',kernel_initializer=real_sp_initializer)(inputs)
    y = Conv1D(filters=int(J*Q),kernel_size=int(T),padding='valid',strides=10, activation='square_activation',kernel_initializer=imag_sp_initializer)(inputs)
    xy = Add()([x,y])
    #print(xy)
    #c1 = Conv1D(24,128,activation='relu',strides=1,padding='valid')(xy)
    #p1 = MaxPooling1D(pool_size=2,strides=1, padding='valid')(c1)
    #d1 = Dropout(0.2)(p1)
    c2 = Conv1D(128,4,activation='relu',strides=10,padding='valid')(xy)
    #p2 = MaxPooling1D(pool_size=100,strides=10, padding='valid')(c2)
    d2 = Dropout(0.2)(c2)
    c3 = Conv1D(128,4,activation='relu',strides=10,padding='valid')(d2)
    #print(c3)
    p3 = MaxPooling1D(pool_size=10,strides=5, padding='valid')(c3)
    #print(p3)
    #d3 = Dropout(0.1)(p3)
    #print(d3)
    #c4 = Conv1D(4,16,activation='relu',strides=1,padding='valid')(d2)
    f1 = Flatten()(p3)
    #print(f1)
    dn1 = Dense(128,activation='sigmoid')(f1)
    dn2 = Dense(32,activation='sigmoid')(dn1)
    #d4 = Dropout(0.2)(dn1)
    predictions = Dense(2,activation='softmax')(dn2)
    
    #training and evaluating the model
    model = Model(inputs=inputs, outputs=predictions)
    return model

#%%
#important variables
Epochs = 1000;
Nsplits = 10;
batch=10;
avg = 0;
tsind = []; #for book-keeping of which samples are in training and test on each fold
Kn = Nsplits;
N,J,Q,S    = 50,2,256,96
lr = 0.001 #learning rate
"""N:int, length of the smallest (highest frequency) filter >=S
J: int, number of octave to decompose
Q: int, number of filters per octave
S: int, number of spline regions"""

#finding the T
myfil = tf_hermite_complex(S=S,deterministic=0,renormalization=np.amax,initialization='gabor',chirplet=1);
[_,_,T] = create_filter_banks_complex(filter_class=myfil,N=N,J=J,Q=Q);
print ('T: ' + str(T))

#%%
#defining my model using KERAS Functional API
model = spline_model(J = J, Q = Q, T = T)
model.summary()

#%%

y_org = le.fit_transform(Y_train.values)
class_weights = class_weight.compute_class_weight('balanced',\
                                np.unique(y_org),y_org)

#%%
#the spline filters are present in the layers x and y
#rmsprop = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.1)
#with tf.Session(config=tf.ConfigProto(
#                    intra_op_parallelism_threads=96)) as sess:
#    sess.run(tf.global_variables_initializer())
#    K.set_session(sess)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience = 5, mode='auto', baseline=None, restore_best_weights=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
csv_logger = CSVLogger('Spline_log\\spline_log.csv', append=False, separator=',')
model.fit(X_train,label_tr,batch_size=32,\
    epochs=Epochs,validation_data=(X_val,label_va)\
    ,callbacks=[csv_logger,early_stopping],class_weight=class_weights)
scr,acc=model.evaluate(X_test,label_te,batch_size=batch)
model.save_weights('Spline_log\\spline_weights.h5')
avg=avg+acc
print(' Test accuracy: '+str(acc))

#%%
y_pred = model.predict(X_test)
print(1-((np.abs(np.ravel(np.round(y_pred)) - label_te)).sum()/len(label_te)))

#%%
Y_test_pat = Y_test[~Y_test.index.duplicated(keep='first')]
temp = y_pred.reshape(int(len(y_pred)/k),k)
#%%
temp1,_ = mode(np.round(temp).T)
temp1 = temp1.T

#%%
print(1-(np.abs(np.ravel(temp1) - le.fit_transform(Y_test_pat.values))).sum()/len(temp1))

#%% VERY IMPORTANT FOR CHECKING FILTER BANKS
myfil = tf_hermite_complex(S=45,deterministic=0,renormalization=\
                           np.amax,initialization='gabor',chirplet=1);
[real_bank,imag_bank,T] = create_filter_banks_complex(filter_class=\
                            myfil,N=50,J=2,Q=256);
print(real_bank.shape)
n = np.linspace(-1,1,num=real_bank.shape[1])
plt.plot(n,np.abs(np.fft.fftshift(np.fft.fft(real_bank[0,:]))))
plt.plot(n,np.abs(np.fft.fftshift(np.fft.fft(real_bank[255,:]))))
plt.plot(n,np.abs(np.fft.fftshift(np.fft.fft(real_bank[511,:]))))
plt.legend(['0','255','511'])
plt.show()