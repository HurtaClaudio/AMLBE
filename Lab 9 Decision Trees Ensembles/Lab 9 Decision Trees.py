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

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1)

clf.fit(X_train,y_train)

tree.export_graphviz(clf,out_file='tree.dot')
## se puede visualizar copiando y pegando el contenido del archivo .dot en http://www.webgraphviz.com/

## importancia de las variables en el árbol
feature_importance=clf.feature_importances_

## predicciones y probabilidades en la muestra de entrenamiento
pred_train=clf.predict(X_train)              #predicción
probs_train=clf.predict_proba(X_train)          #probabilidad

## qué regla o nodo se activa para cada predicción?
rules=clf.apply(X_train)

## utilizando esta función puedo escribir el árbol como reglas de decisión (lenguaje humano)

from sklearn.tree import _tree
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print "def tree({}):".format(", ".join(feature_names))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print "{}if {} <= {}:".format(indent, name, threshold)
            recurse(tree_.children_left[node], depth + 1)
            print "{}else:  # if {} > {}".format(indent, name, threshold)
            recurse(tree_.children_right[node], depth + 1)
        else:
            print "{}return {}".format(indent, tree_.value[node])

    recurse(0, 1)
    
# la función debe ejecutarse desde el terminal: tree_to_code(clf,X_train.columns)

#calculemos el ajuste accuracy del modelo en train y en test

probs2=clf.predict_proba(X_test)
#una vez que ya tengo las predicciones puedo calcular matriz de confusión, accuracy, recall, precision, auc, etc etc

score_train = clf.score(X_train, y_train) 
score_testing = clf.score(X_test, y_test) 

#para calcular otras métricas precision, recall, etc necesito también predecir en testing
   
pred_testing=clf.predict(X_test)               #predigo en base testing

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









