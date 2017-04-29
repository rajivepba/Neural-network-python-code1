# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 20:26:48 2016

@author: Gunnvant
"""
import pandas as pd
from pandas import DataFrame,Series
import sklearn.ensemble as ensemble
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir('F:\\Work\\Python\\WNS Python\\Machine Learning Training\\Class Codes and Data\\Day 2\\RandomForests-ML')

data=pd.read_csv('Building.csv')

X=data.drop(['Y1','Y2'],axis=1)
y=data.Y1

##Splitting data into test and train samples
import sklearn.cross_validation as cross_validation
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

##Building a base model
reg=ensemble.RandomForestRegressor(oob_score=True,n_estimators=100)
reg.fit(X_train,y_train)

print reg.oob_score_

Importance=DataFrame({'Importance':reg.feature_importances_,"Columns":X_train.columns})
Importance.sort_values("Importance",ascending=False,inplace=True)
Importance.plot('Columns',"Importance",kind='bar')

##Base performance
import sklearn.metrics as metrics
metrics.explained_variance_score(y_test,reg.predict(X_test))

##Tune the forest.....same as classifier