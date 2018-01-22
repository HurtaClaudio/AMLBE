# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:19:09 2017

@author: David Diaz
"""

import pandas as pd #panda dataframes
import numpy as np

Location = r'Tabla_impago_full.xlsx'

#importo tabla 

Tabla1 = pd.read_excel(Location,sheet_name="Tabla_B")

#leo tabla ignoro los IDs
data = Tabla1.iloc[:,2:]
data=data.fillna(data.mean())

data1=data.iloc[:,0:2]

#usando toda la data encontraré los vecinos más cercanos de cada fila (empresa)

from sklearn.neighbors import NearestNeighbors

#encuentro los k=2 vecinos más cercanos para cada fila en la data
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data1)

# sklearn me devuelve el índice de los k vecinos más cercanos y la distancia promedio
distances, indices = nbrs.kneighbors(data1)

# además puedo pedir que me cree una matriz de adyacencia (vecinos más cercanos)
matrizvecinos=nbrs.kneighbors_graph(data1).toarray()



## ahora ocuparé esta estrategia de los k-vecinos más cercanos para crear un modelo predictor de clase
#separo la variable dependiente de las explicativas
Xs=data.iloc[:,1:]
Y=data.iloc[:,0]


## separo mi base de datos en dos muestras, una muestra de entrenamiento y una de prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xs, Y, test_size=0.5, random_state=42)

#el comando KDTree leerá la base de datos de entrenamiento, y está será utilizada como mi "modelo"
#es decir cada vez que quiera predecir algo en la base de datos test, modeloknn buscará los más parecidos en la base training y
# hará una proyección en base a eso

from sklearn.neighbors import KDTree
#leo la base X_Train
modeloknn = KDTree(X_train.values, leaf_size=2)


distancias= np.empty((0,2))
indices= np.empty((0,2))

## y luego para cada fila en la base Train le voy buscando sus 2 vecinos más cercanos

counter=0
for row in X_test.values:
    counter=counter+1
    print counter
    row.reshape(1, -1)
    dist, indx = modeloknn.query(row, k=2)     
    distancias=np.vstack([distancias,dist])
    indices=np.vstack([indices,indx])
    
###
## sklearn tiene una librería que realiza este proceso de manera automática, es decir, me permite predecir la clase de un ejemplo
## usando como "modelo" los k-vecinos más cercanos de una base de entrenamiento.
    
## importaré la libreria de KNeighborsClassifier y las librerias que me permiten medir el ajuste del clasificador
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


#configuro el modelo
clf_knn=KNeighborsClassifier(2)

#lo entreno en la base train
clf_knn.fit(X_train,y_train)

#una vez entrenado primero lo ocupo para predecir en la base train
y_pred_train=clf_knn.predict(X_train)


#**en la base train, los valores de los k-vecinos más cercanos a una fila de esa misma base se ocupan para hacer la predicción**

#una vez que ya tengo las predicciones puedo calcular matriz de confusión, accuracy, recall, precision, auc, etc etc
cm_train=confusion_matrix(y_train,y_pred_train)

acc_train=metrics.accuracy_score(y_train,y_pred_train)
accuracy_train=clf_knn.score(X_train,y_train)

#calculo recall, precision, auc, fpr, tpr, etc en ambas bases
fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, y_pred_train, pos_label=1)
auc_train=metrics.auc(fpr_train, tpr_train)
recalls_train=recall_score(y_train, y_pred_train, average=None)
preciss_train=precision_score(y_train, y_pred_train, average=None)
f1sc_train=f1_score(y_train, y_pred_train, average=None)


##ahora puedo usar el modelo para predecir en testing
## cada vez que quiera predecir algo en la base de datos test, clf_knn buscará los más parecidos en la base training y
# hará una proyección en base a eso

y_pred_test=clf_knn.predict(X_test)

#una vez que ya tengo las predicciones puedo calcular matriz de confusión, accuracy, recall, precision, auc, etc etc...

cm_test=confusion_matrix(y_test,y_pred_test)

acc_test=metrics.accuracy_score(y_test,y_pred_test)
accuracy_test=clf_knn.score(X_test,y_test)

fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test, pos_label=1)
auc_test=metrics.auc(fpr_test, tpr_test)
recalls_test=recall_score(y_test, y_pred_test, average=None)
preciss_test=precision_score(y_test, y_pred_test, average=None)
f1sc_test=f1_score(y_test, y_pred_test, average=None)




## finalmente haremos un proceso que usará dos modelos para predecir, uno con KNN y otro con regresión logística
## los resultados de aplicar los modelos en la muestra test quedaran guardados en la variable "resultados_testing"

### configuro y parametrizo los modelos que voy a entrenar
from sklearn.linear_model import LogisticRegression
import pandas as pd

names = ["Nearest Neighbors", "Logistic Regression"]

classifiers = [
    KNeighborsClassifier(2),
    LogisticRegression()]

resultados_testing=[] #genero una matriz vacia donde guardare los scores de los modelos Recall, Precision, AUC, etc

for name, clf in zip(names, classifiers):
    clf.fit(X_train,y_train)
    score_testing = clf.score(X_test, y_test)      #calculo scores en testing
    pred_testing=clf.predict(X_test)               #predigo en base testing

    #calculo recall, precision, auc, fpr, tpr, etc en ambas bases, reporto en vivo
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_testing, pos_label=1)

    recalls=recall_score(y_test, pred_testing, average=None)
    preciss=precision_score(y_test, pred_testing, average=None)
    f1sc=f1_score(y_test, pred_testing, average=None)
    auc1=metrics.auc(fpr, tpr)
    resultados_testing.append((name,score_testing,recalls,preciss,f1sc,auc1))
    colum = ['Name', 'Score', 'Recalls', 'Precision', 'f1sc', 'auc1']
    resultados_testing2 = pd.DataFrame(resultados_testing, columns = colum)
    
    print("Hit Rates/Confusion Matrices en muestra testing:")
    print( resultados_testing2) 
    resultados_testing=[]
    