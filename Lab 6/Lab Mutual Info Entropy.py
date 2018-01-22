# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 15:22:57 2017

@author: David Diaz
"""

import pandas as pd #panda dataframes
import numpy as np #numpy

Location = r'data_simulada.xlsx'

data = pd.read_excel(Location)

from pandas.plotting import scatter_matrix

scatter_matrix(data, alpha=0.4, figsize=(6, 6), diagonal='hist')

correls=data.corr('pearson')

from sklearn.metrics import mutual_info_score

bins=10

x=data['X_rand']
y=data['Y_X_no_lineal']

c_xy = np.histogram2d(x, y, bins)[0]
mi = mutual_info_score(None, None, contingency=c_xy)/np.log(2)

mi_matrix=np.zeros(correls.shape)

for i, column1 in enumerate(data):
    for j, column2 in enumerate(data):
        x=data[column1]
        y=data[column2]
        c_xy = np.histogram2d(x, y, bins)[0]
        mi=mutual_info_score(None, None, contingency=c_xy)/np.log(2)
        mi_matrix[i][j]=mi
       

mi_matrix=pd.DataFrame(data=mi_matrix,columns=correls.columns.values, index=correls.index.values)
mi_matrix_norm=mi_matrix/mi_matrix.max()        


def Entropy(x,bins=0):
    import pandas as pd
    import numpy as np

    df=pd.DataFrame(data=x)
    
      
    if bins>0:
        df['rangos']=pd.cut(df.iloc[:,0],bins)
        count=df['rangos'].value_counts(sort=True)
        probs=count/count.sum()
        logs=np.log2(probs)
        entropy=-probs*logs
        entropy=entropy.sum()
           
    else:
        count=df.iloc[:,0].value_counts(sort=True)
        probs=count/count.sum()
        logs=np.log2(probs)
        entropy=-probs*logs
        entropy=entropy.sum()
    
    return(entropy)
    
Entropy(x,bins=10)



