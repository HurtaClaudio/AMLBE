# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 11:43:28 2017

@author: David Diaz
"""
import pandas as pd #panda dataframes
import numpy as np


from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest


Location = r'Tabla_impago_full.xlsx'

#importo tabla 

Tabla1 = pd.read_excel(Location,sheet_name="Tabla_B")

#leo tabla ignoro los IDs
data = Tabla1.iloc[:,2:]

Xs=data.iloc[:,1:]
Y=data.iloc[:,0]

#relleno los valores perdidos
Xs=Xs.fillna(Xs.mean())

#reescalo las Xs para que funcione PCA y otros algos
scaler = StandardScaler()
scaler.fit(Xs)
Xs_res=scaler = scaler.transform(Xs)
Xs_res=pd.DataFrame(data=Xs_res,index=Xs.index,columns=Xs.columns)

#selección de variables usando Mínima Varianza
cov=Xs_res.cov()
correls=Xs.corr()
sel = VarianceThreshold(threshold=0.01)
filtered1_Xs=sel.fit_transform(Xs)
filtered1_Xs=pd.DataFrame(data=filtered1_Xs,index=Xs.index,columns=Xs.columns)

sel_cols1=sel.get_support(indices=True).T


#reducción de variables usando PCA

pca=PCA(n_components=17)
pca.fit(Xs)
evals=pca.explained_variance_                           # corresponde a los eigenvalues
var_expl=pca.explained_variance_ratio_                  # Varianza explicada por cada componente principal 
evecs=pca.components_.T                                 # corresponde a los eigenvectores
loadings=evecs*np.sqrt(evals)
loadings_filt=np.where(np.abs(loadings)>0.3,loadings,float('nan'))
loadings_filt=pd.DataFrame(data=loadings_filt,index=Xs.columns)


#repito la operación, ahora con n factores que explican %de var total q quiero
pca=PCA(n_components=3)
pca.fit(Xs)
evals=pca.explained_variance_                           # corresponde a los eigenvalues
var_expl=pca.explained_variance_ratio_                  # Varianza explicada por cada componente principal 
evecs=pca.components_.T                                 # corresponde a los eigenvectores
loadings=evecs*np.sqrt(evals)
loadings_filt=np.where(np.abs(loadings)>0.3,loadings,float('nan'))
loadings_filt=pd.DataFrame(data=loadings_filt,index=Xs.columns)
factores=pd.DataFrame(data=pca.fit_transform(Xs),columns=['factor1','factor2','factor3'])   

from pandas.plotting import scatter_matrix
scatter_matrix(factores, alpha=0.4, figsize=(6, 6), diagonal='hist')


## ahora selecciono variables en base a su información mutua respecto de la variable Y
mis=mutual_info_classif(Xs_res,Y)


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
    
EntropiaXs_res=[]

for equis in Xs_res:
    EntropiaXs_res.append((equis,Entropy(Xs_res[equis],10)))

EntropiaXs_Res=pd.DataFrame(data=EntropiaXs_res,columns=['variable','entropia'])
IGR=pd.DataFrame()
IGR['variable']=EntropiaXs_Res['variable']
IGR['IGR']=mis/EntropiaXs_Res['entropia'].values
IGR=IGR.sort_values(by='IGR',ascending=False)


SelectKbest=SelectKBest(mutual_info_classif, k=4)
Xs_sel=SelectKbest.fit_transform(Xs_res,Y)
features = SelectKbest.get_support(indices=True).T

Xs_sel2=Xs[Xs.columns[features]]

## ahora selecciono variables en base a su significancia estadística en una regresión lineal y=f(Xs)

SelectKbest2=SelectKBest(f_classif, k=4)
Xs_sel3=SelectKbest2.fit_transform(Xs_res,Y)
features2 = SelectKbest2.get_support(indices=True).T
Xs_sel3=Xs[Xs.columns[features2]]
