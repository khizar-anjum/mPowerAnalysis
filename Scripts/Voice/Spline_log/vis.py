# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:56:23 2019

@author: Khizar Anjum
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, MaxPooling1D, Dropout, Flatten, Add, Conv1D
from keras.models import Model

#%%
drp = [0.3,0.5,0.7];
files = [pd.read_csv('spline_log'+str(i)+'.csv',index_col=0) for i in drp]

#%%
[plt.plot(df.index.values,df['loss'].values) for df in files]
plt.legend(['dropout = '+str(i) for i in drp])
plt.title('Training loss plot')
plt.show()

#%%
[plt.plot(df.index.values,df['val_loss'].values) for df in files]
plt.legend(['dropout = '+str(i) for i in drp])
plt.title('Validation loss plot')
plt.show()

#%%
[plt.plot(df.index.values,df['sensitivity'].values) for df in files]
plt.legend(['dropout = '+str(i) for i in drp])
plt.title('Sensitivity plot')
plt.show()

#%%
[plt.plot(df.index.values,df['val_sensitivity'].values) for df in files]
plt.legend(['dropout = '+str(i) for i in drp])
plt.title('validaton sensitivity plot')
plt.show()

#%%
def spline_model(J = 2, Q = 128, T = 200):
    inputs = Input(shape=(22050,1))
    #
    #
    x = Conv1D(filters=int(J*Q),kernel_size=int(T),padding='valid',strides=10, activation='square_activation')(inputs)#,kernel_initializer=real_sp_initializer
    #y = Conv1D(filters=int(J*Q),kernel_size=int(T),padding='valid',strides=10, activation='square_activation')(inputs)#,kernel_initializer=imag_sp_initializer
    #xy = Add()([x,y])
    #print(xy)
    #c1 = Conv1D(24,128,activation='relu',strides=1,padding='valid')(xy)
    #p1 = MaxPooling1D(pool_size=2,strides=1, padding='valid')(c1)
    d1 = Dropout(drp)(x)
    c2 = Conv1D(128,4,activation='relu',strides=10,padding='valid')(d1)
    #p2 = MaxPooling1D(pool_size=100,strides=10, padding='valid')(c2)
    d2 = Dropout(drp)(c2)
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
    d4 = Dropout(drp)(dn1)
    dn2 = Dense(32,activation='sigmoid')(d4)
    d5 = Dropout(drp)(dn2)
    predictions = Dense(2,activation='softmax')(d5)
    
    #training and evaluating the model
    model = Model(inputs=inputs, outputs=predictions)
    return model

#%%