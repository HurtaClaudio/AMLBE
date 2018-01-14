#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:57:47 2017

@author: caco
"""

import pandas as pd
import numpy as np

column_names=['A','B','C','D']
date_index=pd.date_range('20130101',periods=6,freq='D')
values=np.random.rand(6,4)

df=pd.DataFrame(values,date_index,column_names)

aux1=df.iloc[2:4,[0,3]]
aux2=df.loc[:,['A','C']]

start_row=2
start_col=2
n=len(df)
m=len(df.columns)
aux3=df.iloc[start_row:n,start_col:m]


col_names2=['ipc','cobre']
date_index2=pd.date_range('20130103',periods=6,freq='D')
values2=np.random.rand(6,2)
df2=pd.DataFrame(values2,date_index2,col_names2)

frames=[df,df2]
df3=pd.concat(frames,axis=1)

aux3=df.dropna()
df5=df3.fillna(method='ffill')
df6=df3.fillna(method='bfill')

#%%

import pandas as pd

location = r'raw_data.xlsx'

df=pd.read_excel(location)
df= df.dropna()
df_norm=(df-df.mean(axis=0))/df.std(axis=0)