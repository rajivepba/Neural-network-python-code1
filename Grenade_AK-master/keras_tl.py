# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 00:56:14 2017

@author: Gunnvant
"""

import os
import numpy as np
import h5py
from PIL import Image
import sklearn.preprocessing as preprocessing
from keras.applications import VGG19,VGG16
from keras.models import Model,load_model
from keras.layers import Dense,GlobalAveragePooling2D,Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array

img_dir='/grenade'
os.chdir(img_dir)
x=[]
for img in os.listdir():
    im=Image.open(img)
    im=im.convert(mode='RGB')
    imrs=im.resize((224,224))
    imrs=img_to_array(imrs)/255
    x.append(imrs)
y=[]
for a in range(len(os.listdir())):
    y.append('grenade')

img_dir='/ak 47'
os.chdir(img_dir)

for img in os.listdir():
    im=Image.open(img)
    im=im.convert(mode='RGB')
    imrs=im.resize((224,224))
    imrs=img_to_array(imrs)/255
    x.append(imrs)

for a in range(len(os.listdir())):
    y.append('ak 47')
x=np.array(x)
y=np.array(y)
enc=preprocessing.LabelEncoder()
y=enc.fit_transform(y)
y=np_utils.to_categorical(y)#1 grenade, 0 ak


## Use VGG 19 as base model to extract features,convolutional layers will remain intact
base_model=VGG19(include_top=False,weights='imagenet',input_shape=(224,224,3),classes=2)
X=base_model.output
X=GlobalAveragePooling2D()(X)
X=Dense(10,activation='relu')(X)
predictions=Dense(2,activation='softmax')(X)
model=Model(inputs=base_model.input,outputs=predictions)
for layer in base_model.layers:
    layer.trainable=False
sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.fit(x,y,batch_size=10,epochs=2)
os.chdir('../models')
model.save_weights('transfer_learning.h5')
model.save('grenade_model.h5')

###Prediction###
os.chdir('..\models')
mod=load_model('grenade_model.h5')
mod.load_weights('transfer_learning.h5')
os.chdir('..\test_images')
im=Image.open('grenade_test.jpeg')
im=im.convert(mode="RGB")
im=im.resize((224,224))
im=np.array(im)
im=im/255.0
import matplotlib.pyplot as plt
plt.imshow(im)
dict(zip(['ak','grenade'],mod.predict(im.reshape(1,224,224,3))[0].tolist()))

####
ak=Image.open('ak_test.jpg')
ak=ak.convert(mode='RGB')
ak=ak.resize((224,224))
ak=np.array(ak)
plt.imshow(ak)
mod.predict(ak.reshape(1,224,224,3))