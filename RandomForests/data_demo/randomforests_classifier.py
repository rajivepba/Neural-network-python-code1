# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:40:38 2016

@author: Gunnvant
"""
import pandas as pd
from pandas import DataFrame,Series
import sklearn.ensemble as ensemble
import os 
import matplotlib.pyplot as plt

os.chdir('E:\\Work\\Python\\WNS Python\\Machine Learning Training\\Class Codes and Data\\Day 2\RandomForests-ML')

##Random forest Classifier
data=pd.read_csv('rock.csv')

X=data.drop('hardness',axis=1)
X=pd.get_dummies(X)
y=data.hardness
y=y.map({'hard':1,'soft':0})

##Splitting data into test training
import sklearn.cross_validation as cross_validation

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

##Building a classifier

clf=ensemble.RandomForestClassifier(oob_score=True)
clf=clf.fit(X_train,y_train)
##Looking at Feature Importances
print clf.feature_importances_
Importance=DataFrame({"Col_name":X_train.columns,'Importance':clf.feature_importances_})
Importance=Importance.sort_values('Importance',ascending=False)
Importance.plot("Col_name","Importance",kind='bar')

##Looking at base performance
import sklearn.metrics as metrics
clf.oob_score_

metrics.accuracy_score(y_test,clf.predict(X_test))

##Look at cross validated scores to get a handle of accuracy
cross_validation.cross_val_score(clf,X_train,y_train,cv=10).mean()

##Tuning model parameters
import sklearn.grid_search as grid_search
clf1=grid_search.GridSearchCV(clf,param_grid=[{"n_estimators":[10,20,50,100,150,300,500,600,700],'criterion':['gini','entropy']}],n_jobs=-1)
clf1=clf1.fit(X_train,y_train)

print clf1.best_params_

print clf1.best_score_

print metrics.accuracy_score(y_test,clf1.predict(X_test))