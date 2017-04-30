#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:08:22 2017

@author: ramius
"""

import os
import pandas as pd
from pandas import DataFrame,Series
import numpy as np
import pickle
import seaborn as sns
import sklearn.feature_selection as feature_selection

##Mutual Information
os.chdir('E:\\Work\\Jigsaw Academy\\Corporate Trainings\\Bocconi\\Batch 3\\Online Sessions\\DRT\\DRT')
data=pd.read_csv('Aquisition_Risk.csv')

data=data.drop(['mths_since_last_delinq','mths_since_last_record'],axis=1)

data=data.dropna()
X=data.drop('Good_Bad',axis=1)
y=data['Good_Bad']
y=y.map({"Good":1,"Bad":0})
X=X.loc[:,X.dtypes!='object']
MI=feature_selection.mutual_info_classif(X,y)
MI=Series(MI,index=X.columns)

'''
with open('MI_Pkl','wb') as f:
    pickle.dump(MI,f)

with open('MI_Pkl') as f:
    mi=pickle.load(f)
'''    
### Information value ###
import information_value as information_value


x=np.array(X['loan_amnt'])
y=np.array(y)
woe=information_value.WOE()
print(woe.woe_single_x(x=woe.discrete(x),y=y))

## MNIST dataset ##
from sklearn import datasets
mnist=datasets.load_digits()

mnist_x=DataFrame(mnist.data)
mnist_y=Series(mnist.target)

'''mnist_x.to_csv('mnist_x.csv',index=False)
mnist_y.to_csv('mnist_y.csv',index=False)'''

import sklearn.linear_model as linear_model
clf=linear_model.LogisticRegressionCV(fit_intercept=True,cv=10,penalty='l1',n_jobs=-1,solver='liblinear')

def get_target(x):
    if x>7:
        return 1
    else:
        return 0

y=mnist_y.map(get_target)

mod=clf.fit(mnist_x,y)
mod.coef_.tolist()

### PCA ###

data=pd.read_csv('nyt.frame.csv')
print(data.columns[np.random.randint(0,data.shape[1],30)])
print(data.loc[np.random.randint(data.index.min(),data.index.max(),5),data.columns[np.random.randint(0,data.shape[1],10)]])
           
### running pca on data ###
X=data.drop('class.labels',axis=1) 
import sklearn.decomposition as decomposition      
pca=decomposition.PCA(n_components=3)
pca_results=pca.fit(X)
loadings=pca_results.components_
loadings_dataframe=DataFrame(loadings,columns=X.columns)
loadings_trans=loadings_dataframe.transpose()
loadings_trans[0].sort_values(ascending=True).head(30)#Notice the difference between results displayed by R, keep in mind eigne vectors talk about 'direction'
loadings_trans[1].sort_values(ascending=True).head(30)

projection=pca_results.transform(X)

'''projection=np.matmul(X,loadings.T)
'''
projection=DataFrame(projection,columns=['pc1','pc2','pc3'])
projection['target']=data['class.labels']
sns.lmplot(x='pc1',y='pc2',hue='target',data=projection,fit_reg=False)
## LDA ###

data=pd.read_csv('authorship.csv')
data['Target']=data.Author.map({'Austen':1,'London':0,'Milton':1,'Shakespeare':1})

X=data.drop(['Author','Target','BookID'],axis=1)
y=data.Target

import sklearn.discriminant_analysis as discriminant_analysis

lda=discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)

mod_lda=lda.fit(X,y)
comp_lda=mod_lda.transform(X)
comp_lda=DataFrame(comp_lda,columns=['comp1'])
comp_lda['row_num']=comp_lda.index
comp_lda['target']=y

sns.lmplot(x='row_num',y='comp1',hue='target',data=comp_lda,fit_reg=False)


pca=decomposition.PCA(n_components=2)
mod_pca=pca.fit(X)
comp_pca=pca.fit(X).transform(X)
comp_pca=DataFrame(comp_pca,columns=['comp1','comp2'])
comp_pca['target']=y

sns.lmplot(x='comp1',y='comp2',hue='target',data=comp_pca,fit_reg=False)


