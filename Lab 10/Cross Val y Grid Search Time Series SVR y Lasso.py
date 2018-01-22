# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:55:17 2017

@author: Raul
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVR
from sklearn import metrics

location = r'TuneoCLP.xlsx'

data=pd.read_excel(location,'Sheet1')
data=data.iloc[-400:,:]

pg={'C':[1,10,100,1000,1000],'kernel':['linear','poly','rbf','sigmoid'],'gamma':[0.1,0.01,0.001,0.0001]}
param_grid=ParameterGrid(pg)
list(param_grid)

col=['r2','y_pred','y_true','res']
hp=[i for i in pg]
columns=col+hp

n_y=1
tw=252
n=len(data)
m=n-tw
q=len(data.columns)
s=len(param_grid)    
out=pd.DataFrame(np.zeros(shape=((m+1)*s,len(columns))),index=[i for i in range(0,(m+1)*s)],columns=columns)
norm=1
count=0
for k in range(0,len(param_grid)):
    model=SVR(kernel=param_grid[k]['kernel'],gamma=param_grid[k]['gamma'],C=param_grid[k]['C'])
    for i in range(0,m+1):
        X_train=data.iloc[i:i+tw,n_y:q]      
        X_test=data.iloc[tw+i-1,n_y:q]
        if norm==1:
            X_mean=X_train.mean(axis=0)
            X_desv=X_train.std(axis=0)
            X_train=(X_train-X_mean)/X_desv
            X_test=(X_test-X_mean)/X_desv
        
        y_train=data.iloc[i:i+tw,0:n_y]
        y_true=data.iloc[tw+i-1,0:n_y]
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test.values.reshape(1,q-n_y))
        res=y_pred-y_true.values
        out.iloc[count,0]=metrics.r2_score(y_train,model.predict(X_train)) # in sample
        out.iloc[count,1]=y_pred
        out.iloc[count,2]=y_true.values
        out.iloc[count,3]=res
        out.iloc[count,4]=param_grid[k]['kernel']
        out.iloc[count,5]=C=param_grid[k]['C']
        out.iloc[count,6]=C=param_grid[k]['gamma']
        count=count+1    
        print "Optimización n° %0.3f de %r" % (count, (m+1)*s)
    
name='SVR Optim USDCLP Curncy.xlsx'   
writer = pd.ExcelWriter(name, engine='xlsxwriter')
out.to_excel(writer, sheet_name='Sheet 1')
writer.save()

#%%


import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import Lasso
from sklearn import metrics

location = r'TuneoCLP.xlsx'

data=pd.read_excel(location ,'Sheet1')
data=data.iloc[-400:,:]


pg={'alpha':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
param_grid=ParameterGrid(pg)

col=['r2','y_pred','y_true','res']
hp=[i for i in pg]
columns=col+hp

n_y=1
tw=252
n=len(data)
m=n-tw
q=len(data.columns)
s=len(param_grid)    
out=pd.DataFrame(np.zeros(shape=((m+1)*s,len(columns))),index=[i for i in range(0,(m+1)*s)],columns=columns)
norm=1
count=0
for k in range(0,len(param_grid)):
    model=Lasso(alpha=param_grid[k]['alpha'])
    for i in range(0,m+1):
        X_train=data.iloc[i:i+tw,n_y:q]      
        X_test=data.iloc[tw+i-1,n_y:q]
        if norm==1:
            X_mean=X_train.mean(axis=0)
            X_desv=X_train.std(axis=0)
            X_train=(X_train-X_mean)/X_desv
            X_test=(X_test-X_mean)/X_desv
        
        y_train=data.iloc[i:i+tw,0:n_y]
        y_true=data.iloc[tw+i-1,0:n_y]
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test.values.reshape(1,q-n_y))
        res=y_pred-y_true.values
        out.iloc[count,0]=metrics.r2_score(y_train,model.predict(X_train)) # in sample
        out.iloc[count,1]=y_pred
        out.iloc[count,2]=y_true.values
        out.iloc[count,3]=res
        out.iloc[count,4]=param_grid[k]['alpha']
        count=count+1    
        print "Optimización n° %0.3f de %r" % (count, (m+1)*s)
    
name='Lasso Optim USDCLP Curncy.xlsx'   
writer = pd.ExcelWriter(name, engine='xlsxwriter')
out.to_excel(writer, sheet_name='Sheet 1')
writer.save()