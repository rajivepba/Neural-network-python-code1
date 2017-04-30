# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 17:20:29 2017

@author: Gunnvant
"""
import os
import numpy as np
from flask import Flask,request
from keras.models import load_model
from PIL import Image
import json

app=Flask(__name__)

@app.route("/")
def index():
        return 'Hi just checking'
@app.route("/predict",methods=['POST'])
def predict():
    os.chdir('../models')
    mod=load_model('grenade_model.h5')
    mod.load_weights('transfer_learning.h5')
    img=Image.open(request.files['file'])
    img=img.convert(mode='RGB')
    img=img.resize((224,224))
    img=np.array(img)
    img=img/255.0
    return json.dumps(dict(zip(['ak','grenade'],mod.predict(img.reshape(1,224,224,3))[0].tolist())))
    


if __name__=='__main__':
    app.run()