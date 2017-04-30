# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 23:48:19 2017

@author: Gunnvant
"""

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
import pandas as pd
from pandas import DataFrame,Series
import numpy as np
import os

'''Model Definition'''

model=Sequential()
model.add(Dense(15,input_shape=(64,)))
model.add(Activation('relu'))
model.add(Dense(10,input_shape=(64,)))
model.add(Activation('relu'))

model.add(Dense(10,activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

''' Data loading '''

os.chdir('../data')

x=pd.read_csv('mnist_x.csv')
y=pd.read_csv('mnist_y.csv',names=['target'])
y=y.target.map({0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'})
y=pd.get_dummies(y)

X=np.array(x)
Y=np.array(y)

model.fit(X,Y,epochs=20,batch_size=128)

model.metrics_names

score = model.evaluate(X,Y)