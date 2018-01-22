# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 17:57:45 2017

@author: David Diaz
"""

import pandas as pd #panda dataframes

Location = r'Tabla_impago_full.xlsx'

#importo tabla 

Tabla1 = pd.read_excel(Location,sheet_name="Tabla_B")

#leo tabla ignoro los IDs
data = Tabla1.iloc[:,2:]
data=data.fillna(data.mean())

data1=data.iloc[:,0:2]

#separo la variable dependiente de las explicativas
Xs=data.iloc[:,1:]
Y=data.iloc[:,0]


## separo mi base de datos en dos muestras, una muestra de entrenamiento y una de prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xs, Y, test_size=0.5, random_state=42)


from imblearn.over_sampling import RandomOverSampler
from collections import Counter
## balanceo muestra de entrenamiento - oversampling
print('Balanceando Muestras ...'.format())+'\n'
print('Original dataset shape {}'.format(Counter(y_train)))+'\n'
ros = RandomOverSampler(random_state=42)
X_reb, y_reb = ros.fit_sample(X_train, y_train)
print('Resampled dataset shape {}'.format(Counter(y_reb)))+'\n'

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


bags=RandomForestClassifier(n_estimators=10, n_jobs=1, verbose=1)

bags.fit(X_reb,y_reb)

estimators=bags.estimators_

tree.export_graphviz(estimators[0],out_file='tree_balanced_1.dot')
tree.export_graphviz(estimators[9],out_file='tree_balanced_10.dot')
## se puede visualizar copiando y pegando el contenido del archivo .dot en http://www.webgraphviz.com/

## predicciones y probabilidades en la muestra de entrenamiento
pred_train=bags.predict(X_train)              #predicción
probs_train=bags.predict_proba(X_train)          #probabilidad


#calculemos el ajuste accuracy del modelo en train y en test

probs2=bags.predict_proba(X_test)
#una vez que ya tengo las predicciones puedo calcular matriz de confusión, accuracy, recall, precision, auc, etc etc

score_train = bags.score(X_train, y_train) 
score_testing = bags.score(X_test, y_test) 

#para calcular otras métricas precision, recall, etc necesito también predecir en testing
   
pred_testing=bags.predict(X_test)               #predigo en base testing

#calculo recall, precision, auc, fpr, tpr, auc, f1 etc en ambas bases
fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, pred_train, pos_label=1)
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, pred_testing, pos_label=1)

auc_train=metrics.auc(fpr_train, tpr_train)
auc_test=metrics.auc(fpr_test, tpr_test)

recalls_train=recall_score(y_train, pred_train, average=None)
recalls_test=recall_score(y_test, pred_testing, average=None)

precis_train=precision_score(y_train, pred_train, average=None)
precis_test=precision_score(y_test, pred_testing, average=None)

f1sc_train=f1_score(y_train, pred_train, average=None)
f1sc_test=f1_score(y_test, pred_testing, average=None)









