# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:35:14 2017

@author: publico
"""

#Laboratorio Clustering K-means

import pandas as pd #panda dataframes
import numpy as np #numpy

from sklearn.preprocessing import StandardScaler #importa libreria sklearn para escalar la data
from sklearn import metrics #importa libreria para evaluar el resultado de los clusters-silueta
from sklearn.cluster import DBSCAN #importa algortimo de kmeans

Location = r'Tabla_impago_full.xlsx'


#importo tabla 

Tabla1 = pd.read_excel(Location,sheet_name="Tabla_B")

data = Tabla1.iloc[:,2:]
stats1=data.describe()

data=data.fillna(data.mean())

data_escalada= StandardScaler().fit_transform(data)

stats2=(pd.DataFrame(data=data_escalada,columns=stats1.columns)).describe()

# Set the parameters by cross-validation
min_samples_range = np.arange(2,10)
eps_range = np.arange(0.1,4,0.1)


grid_results=[]

for min_samples_i in min_samples_range:
    for eps_i in eps_range:
        modelo_clustering = DBSCAN(eps=eps_i,min_samples=min_samples_i).fit(data_escalada)
        etiquetas=modelo_clustering.labels_
        n_clusters_ = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
        if n_clusters_<2:
            silueta=0
            ch=0
        else:
            silueta=metrics.silhouette_score(data_escalada, modelo_clustering.labels_,metric='euclidean')
            ch=metrics.calinski_harabaz_score(data_escalada, modelo_clustering.labels_)
        #print(eps_i,min_samples_i,silueta,ch,n_clusters_)
        grid_results.append((min_samples_i,eps_i,silueta,ch,n_clusters_))
        
#genero el DF con los resultados
resultados=pd.DataFrame(data=grid_results,columns=['min_samples','eps','silueta','ch','n_cluster'])
resultados.to_excel('resultados_dbscan.xlsx')

print(resultados['silueta'].max())

def heat_map(x,y,z,x_name,y_name,z_name):
    import numpy as np
    import matplotlib.pyplot as plt
    x=np.unique(x)
    y= np.unique(y)   
    z = np.array(z).reshape(len(y), len(x))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(z, interpolation='nearest', cmap=plt.cm.coolwarm)#
    title=str(z_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.colorbar()
    plt.xticks(np.arange(len(x)), x, rotation=0)
    plt.yticks(np.arange(len(y)), y)
   # for i in range(0,len(y)):
   #     for j in range(0,len(x)):
   #         plt.text(j,i, '%.3f' % z[i, j],ha="center", va="center")
    plt.title(title)
    plt.show()

heat_map(resultados['min_samples'],resultados['eps'],resultados['silueta'],'min_sample','eps','silueta')


#%% DBSCAN con parametros optimizados

#ahora me gustaría generar una función que haga el clustering por mi    
import pandas as pd #panda dataframes

from sklearn.preprocessing import StandardScaler #importa libreria sklearn para escalar la data
from sklearn import metrics
from sklearn.cluster import DBSCAN

#Location = r'D:\Dropbox\FEN\Semestres\2017-2\AMLBE\Nuevos Laboratorios en Python\Lab 3 Clustering\Tabla_impago_full.xlsx'
Location = r'Tabla_impago_full.xlsx'
Tabla1 = pd.read_excel(Location,sheet_name="Tabla_B")

data = Tabla1.iloc[:,2:]
#defino mi función que necesita dos inputs: la data donde quiero clusterizar, y el número de K
def clusteriza(tabla,epsilon,minsamples):
    data=tabla
    data=data.fillna(data.mean())
    data_escalada= StandardScaler().fit_transform(data)
    modelo_clustering = DBSCAN(eps=epsilon,min_samples=minsamples).fit(data_escalada)
    etiquetas=modelo_clustering.labels_
    n_clusters_ = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    
    silueta=metrics.silhouette_score(data_escalada, modelo_clustering.labels_,metric='euclidean')
    ch=metrics.calinski_harabaz_score(data_escalada, modelo_clustering.labels_)
    
    #la función me devuelve los resultados
    return (etiquetas,silueta, ch,n_clusters_)

#################################################################################################
#aquí por ejemplo ejecuto la función sobre matriz data con K=
etiqs,sils,CH, nclusters=clusteriza(data,3,2)
#################################################################################################
df_eti=pd.DataFrame(data=etiqs,columns=['cluster'])
cdata=pd.concat([df_eti,data],axis=1)

#creo la matriz de centroides para cada cluster, ie. la media por cluster
centroides1=cdata.groupby('cluster').mean()
centroides1=centroides1.reset_index(drop=False)

from pandas.plotting import parallel_coordinates

#no dibuje todas las columnas porque el gráfico queda muy grande

#dibujo sólo los centroides de la data escalada para que sea más fácil interpretar:
parallel_coordinates(centroides1.iloc[:,0:10], 'cluster',colormap='gist_rainbow')