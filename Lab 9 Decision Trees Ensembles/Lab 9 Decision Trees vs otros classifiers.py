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



# ahora ocuparé esta estrategia de los k-vecinos más cercanos para crear un modelo predictor de clase
#separo la variable dependiente de las explicativas
Xs=data.iloc[:,1:]
Y=data.iloc[:,0]


## separo mi base de datos en dos muestras, una muestra de entrenamiento y una de prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xs, Y, test_size=0.5, random_state=42)

from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

## balanceo muestra de entrenamiento - oversampling

print('Balanceando Muestras ...'.format())+'\n'
print('Original dataset shape {}'.format(Counter(y_train)))+'\n'
ros = RandomOverSampler(random_state=42)
X_reb, y_reb = ros.fit_sample(X_train, y_train)
print('Resampled dataset shape {}'.format(Counter(y_reb)))+'\n'


## los resultados de aplicar los modelos en la muestra test quedaran guardados en la variable "resultados_testing"

### configuro y parametrizo los modelos que voy a entrenar
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


names = ["Logistic Regression","Nearest Neighbors", 'Simple DT', 'RandomForest','Adaboost']

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(10),
    DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1),
    RandomForestClassifier(n_estimators=100, n_jobs=1, verbose=1),
    AdaBoostClassifier(n_estimators=100)]

resultados_testing=[] #genero una matriz vacia donde guardare los scores de los modelos Recall, Precision, AUC, etc


for name, clf in zip(names, classifiers):
    clf.fit(X_reb,y_reb)
    score_testing = clf.score(X_test, y_test)      #calculo scores en testing
    pred_testing=clf.predict(X_test)               #predigo en base testing

    #calculo recall, precision, auc, fpr, tpr, etc en ambas bases, reporto en vivo
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_testing, pos_label=1)
       
    recalls=recall_score(y_test, pred_testing, average=None)
    preciss=precision_score(y_test, pred_testing, average=None)
    f1sc=f1_score(y_test, pred_testing, average=None)
    auc1=metrics.auc(fpr, tpr)
    resultados_testing.append((name,score_testing,recalls,preciss,f1sc,auc1))


resultados_testing=pd.DataFrame(data=resultados_testing, columns=['Modelo','Accuracy','recall','precision','fscore','auc'])