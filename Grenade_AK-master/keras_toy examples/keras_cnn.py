# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 23:00:44 2017

@author: Gunnvant
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense,Flatten
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

os.chdir('../data')
data=pd.read_csv('mnist.csv')
y=data.label
x=data.drop('label',axis=1)

## Train test split
X_train,X_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=100)
## Conversion to numpy arrays
X_train=np.array(X_train)
X_test=np.array(X_test)
##Scaling images
X_train=X_train/255.0
X_test=X_test/255.0
##Reshaping: nrows,28,28,1 channel (images are greyscale)
x_train=X_train.reshape(X_train.shape[0],28,28,1)
x_test=X_test.reshape(X_test.shape[0],28,28,1)
##Visualizing the images
plt.imshow(x_train[0,:,:].reshape(28,28),cmap='gray')
for i in range(20):
   plt.imshow(x_train[i,:,:].reshape(28,28),cmap='gray')
   plt.show()
##One hot encoding the target 
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
##LeNet model configuration(Original LeNet expects 32*32 input, I have modified the implimentation)
model=Sequential()
model.add(Conv2D(filters=6,kernel_size=(3,3),padding='same',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16,kernel_size=(3,3),padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
print(model.summary())

model.fit(x_train,y_train,epochs=10,batch_size=64)
model.metrics_names
score=model.evaluate(x_test,y_test)s
results=model.predict_proba(x_test)
results[1,:]
plt.imshow(x_test[1,:,:].reshape(28,28),cmap='gray')