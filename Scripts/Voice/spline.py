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

import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import mode
from keras.callbacks import CSVLogger#, EarlyStopping
from keras.layers import Input, Dense, MaxPooling1D, Dropout, Flatten,\
BatchNormalization, Activation #Add,
from keras.models import Model
from keras.layers.convolutional import Conv1D
from sklearn.utils import class_weight
from keras.utils.generic_utils import get_custom_objects    
from spline_functions import k_selections, tf_hermite_complex,\
 real_sp_initializer, create_filter_banks_complex, sensitivity, specificity
from imblearn.over_sampling import ADASYN

#%% Using k_selections
mfccSeries = pd.read_csv('..\\..\\Data\\VoiceData\\splineData\\SeriesX_spline.csv')
X_train = np.load('..\\..\\Data\\VoiceData\\splineData\\X_spline_train.npy')
X_val = np.load('..\\..\\Data\\VoiceData\\splineData\\X_spline_val.npy')
X_test = np.load('..\\..\\Data\\VoiceData\\splineData\\X_spline_test.npy')

k = 4;
X_train, Y_train = k_selections(k, X_train, mfccSeries, 'train')
X_val, Y_val = k_selections(k, X_val, mfccSeries, 'val')
X_test, Y_test = k_selections(k, X_test, mfccSeries, 'test')
#%%
le = preprocessing.LabelEncoder()
label_tr = to_categorical(le.fit_transform(Y_train.values),num_classes=2)
label_va = to_categorical(le.fit_transform(Y_val.values),num_classes=2)
label_te = to_categorical(le.fit_transform(Y_test.values),num_classes=2)

#X_train = np.expand_dims(X_train,axis=2)
#X_test = np.expand_dims(X_test,axis=2)
#X_val = np.expand_dims(X_val,axis=2)

#%%
def square_activation(x):
    return K.square(x)

get_custom_objects().update({'square_activation': Activation(square_activation)})

#%% Making model in here!
def spline_model(J = 2, Q = 128, T = 200):
    inputs = Input(shape=(22050,1))
    x = Conv1D(filters=int(J*Q),kernel_size=int(T),strides=50,padding='valid'\
               )(inputs)#=real_sp_initializer),kernel_initializer = 'glorot_normal'
    b1 = BatchNormalization()(x)
    d1 = Dropout(0.2)(b1)
    #c2 = Conv1D(128,4,activation='relu',strides=10,padding='valid')(d1)
    #d2 = Dropout(0.3)(c2)
    #c3 = Conv1D(128,4,activation='relu',strides=10,padding='valid')(d2)
    p3 = MaxPooling1D(pool_size=20, padding='valid')(d1)#,kernel_initializer = 'glorot_normal'
    f1 = Flatten()(p3)
    dn1 = Dense(128,activation='relu')(f1)
    d4 = Dropout(0.2)(dn1)
    dn2 = Dense(32,activation='relu')(d4)
    d5 = Dropout(0.2)(dn2)
    predictions = Dense(2,activation='softmax')(d5)

    model = Model(inputs=inputs, outputs=predictions)
    
    return model

#%%
#important variables
Epochs = 500;
batch=10;
avg = 0;
tsind = []; #for book-keeping of which samples are in training and test on each fold
N,J,Q,S    = 50,2,64,190
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

#y_org = le.fit_transform(Y_train.values)
#class_weights = class_weight.compute_class_weight('balanced',\
#                                np.unique(y_org),y_org)
sm = ADASYN(random_state=42)
X_res, y_res = sm.fit_resample(np.squeeze(X_train), le.fit_transform(Y_train.values))
label_tr = to_categorical(y_res,num_classes=2)
#%%
#the spline filters are present in the layers x and y
#rmsprop = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.1)
#with tf.Session(config=tf.ConfigProto(
#                    intra_op_parallelism_threads=96)) as sess:
#    sess.run(tf.global_variables_initializer())
#    K.set_session(sess)
#early_stopping = EarlyStopping(monitor='val_sensitivity', min_delta=0, patience = 5, mode='auto', baseline=None, restore_best_weights=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[sensitivity, specificity])
csv_logger = CSVLogger('Spline_log\\spline_log'+str(0.5)+'.csv', append=False, separator=',')
model.fit(X_res.reshape((-1,22050,1)),label_tr,batch_size=32,\
    epochs=Epochs,validation_data=(X_val.reshape((-1,22050,1)),label_va)\
    ,callbacks=[csv_logger])#,early_stopping],class_weight=class_weights
acc,sens,spec=model.evaluate(X_test.reshape((-1,22050,1)),label_te,batch_size=batch)
model.save_weights('Spline_log\\spline_weights'+str(0.5)+'.h5')
print('Test sensitivity: '+str(sens))
print('Test specificity: '+str(spec))
#%%
y_pred = model.predict(X_test.reshape((-1,22050,1)))
print(1-((np.abs(np.round(y_pred) - label_te)[:,0]).sum()/len(label_te)))
#%%
Y_test_pat = Y_test[~Y_test.index.duplicated(keep='first')]
temp = y_pred[:,1].reshape(int(len(y_pred[:,1])/k),k)
#%%
temp1,_ = mode(np.round(temp).T)
temp1 = temp1.T
print(1-(np.abs(np.ravel(temp1) - le.fit_transform(Y_test_pat.values))).sum()/len(temp1))
#%% CODE FOR PLOTTING
history = pd.read_csv('Spline_log\\spline_weights'+str(0.5)+'.csv')
spline1 = np.squeeze(model.get_weights()[0])

#%% VERY IMPORTANT FOR CHECKING FILTER BANKS
myfil = tf_hermite_complex(S=190,deterministic=0,renormalization=\
                           np.amax,initialization='gabor',chirplet=1);
[real_bank,imag_bank,T] = create_filter_banks_complex(filter_class=\
                            myfil,N=50,J=2,Q=128);
print(real_bank.shape)
m = np.linspace(-1,1,num=real_bank.shape[1])
plt.plot(m,np.abs(np.fft.fftshift(np.fft.fft(real_bank[80,:]))))
plt.plot(m,np.abs(np.fft.fftshift(np.fft.fft(real_bank[122,:]))))
plt.plot(m,np.abs(np.fft.fftshift(np.fft.fft(real_bank[255,:]))))
plt.legend(['80','122','255'])
plt.show()

#%% COMPARISON
Fs = 2205;
l = 170 #filter number to view
m = np.linspace(-Fs/2,Fs/2,num=real_bank.shape[1])
n = np.linspace(-Fs/2,Fs/2,num=spline1.shape[0])
plt.plot(m,np.abs(np.fft.fftshift(np.fft.fft(real_bank[l,:]))))
plt.plot(n,np.abs(np.fft.fftshift(np.fft.fft(spline1[:,l]))))
plt.legend(['initialization','after learning'])
plt.show()
