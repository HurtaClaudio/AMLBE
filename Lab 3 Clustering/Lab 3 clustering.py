# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:35:14 2017

@author: publico
"""

#Laboratorio Clustering K-means

import pandas as pd #panda dataframes

from sklearn.preprocessing import scale #importa libreria sklearn para escalar la data
from sklearn import metrics #importa libreria para evaluar el resultado de los clusters-silueta
from sklearn.cluster import KMeans #importa algortimo de kmeans

#Location = r'D:\Dropbox\FEN\Semestres\2017-2\AMLBE\Nuevos Laboratorios en Python\Lab 3 Clustering\Tabla_impago_full.xlsx'
Location = r'Tabla_impago_full.xlsx'


#importo tabla 

Tabla1 = pd.read_excel(Location,sheet_name="Tabla_B")

#leo tabla ignoro los IDs
data = Tabla1.iloc[:,2:]

#estadistica descriptiva de la data original
stats1=data.describe()

#relleno los valores perdidos
data=data.fillna(data.mean())

#reescalo toda la data - necesario para un buen clustering
data_escalada=scale(data)

#chequeo estadística descriptiva de la data reescalada
stats2=(pd.DataFrame(data=data_escalada,columns=stats1.columns)).describe()

#creo mi modelo de clustering kmeans k=5 en la data escalada
modelo_clustering = KMeans(n_clusters=5).fit(data_escalada)

#devuelvo las etiquetas de pertenencia a clusters para cada fila
etiquetas=modelo_clustering.predict(data_escalada)

#evalúo el clustering - silueta y CH
silueta=metrics.silhouette_score(data_escalada, modelo_clustering.labels_,metric='euclidean')
ch=metrics.calinski_harabaz_score(data_escalada, modelo_clustering.labels_)

#muestro métricas a pantalla
print(silueta)
print(ch)


#ahora me gustaría encontrar el número óptimo de Ks
#two-step casero x-means

#genero matrices vacias donde guardare los resultados de probar muchos Ks

Ks=[]
siluetas=[]
CHs=[]


#genero un for que probara desde K=2 hasta K=10
for K in range(2,11):
    modelo_clustering = KMeans(n_clusters=K).fit(data_escalada)
    silueta=metrics.silhouette_score(data_escalada, modelo_clustering.labels_,metric='euclidean')
    ch=metrics.calinski_harabaz_score(data_escalada, modelo_clustering.labels_)
    
    #voy guardando el K, silueta y CH
    Ks.append(K)
    siluetas.append(silueta)
    CHs.append(ch)
    #muestro resultado en cada iteración
    print(K,silueta,ch)
    

#guardo los resultadods como DFs
DF_Ks=pd.DataFrame({'Ks': Ks})
DF_Siluetas=pd.DataFrame({'Siluetas': siluetas})
DF_CH=pd.DataFrame({'Calinsky-Harabaz': CHs})

#concateno en un sólo DF resultado
resultado=pd.concat([DF_Ks,DF_Siluetas,DF_CH],axis=1)

#muestro gráficamente en un gráfico de codo o elbow los resultados
resultado.plot(x='Ks', y='Siluetas')
resultado.plot(x='Ks',y='Calinsky-Harabaz')


#%%

#ahora me gustaría generar una función que haga el clustering por mi    
import pandas as pd #panda dataframes

from sklearn.preprocessing import StandardScaler #importa libreria sklearn para escalar la data
from sklearn import metrics #importa libreria para evaluar el resultado de los clusters-silueta
from sklearn.cluster import DBSCAN #importa algortimo de kmeans

#Location = r'D:\Dropbox\FEN\Semestres\2017-2\AMLBE\Nuevos Laboratorios en Python\Lab 3 Clustering\Tabla_impago_full.xlsx'
Location = r'Tabla_impago_full.xlsx'
Tabla1 = pd.read_excel(Location,sheet_name="Tabla_B")

data = Tabla1.iloc[:,2:]

#defino mi función que necesita dos inputs: la data donde quiero clusterizar, y el número de K
def clusteriza(tabla,K):
    data=tabla
    stats1=data.describe()
    data=data.fillna(data.mean())
    data_escalada=scale(data)
    stats2=(pd.DataFrame(data=data_escalada,columns=stats1.columns)).describe()
    modelo_clustering = KMeans(n_clusters=K).fit(data_escalada)
    etiquetas=modelo_clustering.predict(data_escalada)
    
    silueta=metrics.silhouette_score(data_escalada, modelo_clustering.labels_,metric='euclidean')
    ch=metrics.calinski_harabaz_score(data_escalada, modelo_clustering.labels_)
    
    df_eti=pd.DataFrame(data=etiquetas,columns=['cluster'])
    df_cs=pd.DataFrame(data=data_escalada,columns=data.columns)        
    
    c_data=pd.concat([df_eti,data],axis=1)
    csdata=pd.concat([df_eti,df_cs],axis=1)    
    
    print(K,silueta)

    #la función me devuelve las matrices de estadísticas, etiquetas, silueta, ch, data sin escalar, data escalada
    return (stats1, stats2, etiquetas,silueta, ch, c_data,csdata)

#################################################################################################
#aquí por ejemplo ejecuto la función sobre matriz data con K=
des1,des2,klusters,sils,CH, cdata,csdata=clusteriza(data,6)
#################################################################################################


#guardo la data con la columna cluster
cdata.to_excel('data_con_etiqueta.xlsx')

#creo la matriz de centroides para cada cluster, ie. la media por cluster
centroides1=cdata.groupby('cluster').mean()
centroides1=centroides1.reset_index(drop=False)

#lo mismo pero para data escalada
centroides2=csdata.groupby('cluster').mean()
centroides2=centroides2.reset_index(drop=False)

#genero un lindo gráfico de coordenadas paralelas
#este tipo de gráficos me permite ver como es cada variable separando por cluster

from pandas.plotting import parallel_coordinates

#no dibuje todas las columnas porque el gráfico queda muy grande

#dibujo sólo los centroides de la data escalada para que sea más fácil interpretar:
parallel_coordinates(centroides1.iloc[:,0:10], 'cluster',colormap='gist_rainbow')

#puedo ver el detalle de la diferencia por cluster mirando un gráfico de cajas
cdata.boxplot('4_LnVentas_',by='cluster',figsize=(12, 8))


#serán estas diferencias estadísticamente significativas? son los clusters realmente diferentes unos de otros?

#para eso necesito hacer un test - ANOVA

#importo stats desde scipy que me permite hacer tests estadísticos
from scipy import stats

#devuelve F y p-value
F, p = stats.f_oneway(cdata['4_LnVentas_'], cdata['cluster']) 

print(F,p)  


#me gustaría testear todas las columnas de una sola vez
#genero un for que ejecute el test columna a columna

columnas=data.columns._data
x=0
anovas=[]

#para las columnas 1 al 19 en cdata testeo anova por cluster
for cols in cdata.iloc[:,1:19].columns:
    cdata.boxplot(cols,by='cluster',figsize=(12, 8))
    F, p = stats.f_oneway(cdata[cols], cdata['cluster']) 
    
    #voy guardando mi resultado en la matriz anovas
    anovas.append((columnas[x],F,p))
    x=x+1

#genero el DF con los resultados
resultados=pd.DataFrame(data=anovas,columns=['Columna','F','p'])
resultados.to_excel('resultados_anova.xlsx')
 