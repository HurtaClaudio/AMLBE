# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 16:26:25 2017

@author: David Diaz
"""

import pandas as pd #panda dataframes
import numpy as np #numpy
import scipy.stats as sci


Location = r'buy_computer.xlsx'

data = pd.read_excel(Location)
data=data.astype('str')

x=data['class:buy_computer']
y=data['student']

cross_tab=pd.crosstab(x,y)

c, p, dof, expected =sci.chi2_contingency(cross_tab)


from sklearn.metrics import mutual_info_score

mi = mutual_info_score(None, None, contingency=cross_tab)/np.log(2)

fils,cols=data.shape

mi_matrix=np.zeros(shape=(cols,cols))
chi2_p_values_matrix=np.ones(shape=(cols,cols))

for i, column1 in enumerate(data):
    for j, column2 in enumerate(data):
        x=data[column1]
        y=data[column2]
        mi=mutual_info_score(None, None, contingency=cross_tab)
        mi_matrix[i][j]=mi
                
        if i!=j:
            cross_tab=pd.crosstab(x,y)
            c, p, dof, expected =sci.chi2_contingency(cross_tab)
            chi2_p_values_matrix[i][j]=p
            

chi2_p_values_matrix=pd.DataFrame(data=chi2_p_values_matrix,columns=data.columns.values, index=data.columns.values)

            
mi_matrix=pd.DataFrame(data=mi_matrix,columns=data.columns.values, index=data.columns.values)
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
    
Entropy(x)
