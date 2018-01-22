# -*- coding: utf-8 -*-
"""

@author: Raul
"""

#%%

# 1. Importar la base de datos "data_clasificación.xlsx"


import pandas as pd

location = r'data_clasificacion.xlsx'
data=pd.read_excel(location)
#El archivo excel original se llamaba "data_clasificasion.xlsx"

#%%

# 2. Estime un modelo de regresión lineal donde la variable dependiente sea el tipo de cambio
# Utilizando el resto de las variables como explicativas, utilice el alguna métrica
# para evaluar el ajuste de su regresión



# Con sklearn 

from sklearn.linear_model import LinearRegression
from sklearn import metrics

y=data.iloc[:,0]
X=data.iloc[:,1:]

params=[]
model=LinearRegression()
model.fit(X,y)
params.append(model.intercept_)
params.append(model.coef_)
y_est=model.predict(X)
metrics.mean_absolute_error(y,y_est)
metrics.r2_score(y,y_est)

# Con StatsModels

import numpy as np
import statsmodels.formula.api as smf
X_b=np.c_[np.ones((len(X),1)),X]
est = smf.OLS(y,X_b).fit()
est.summary() 

#%%

# 3. Calcule los retornos logaritmicos para toda la base y agruegue una columna auxiliar donde transforme
# el retorno del tipo de cambio a una variable binaria 0, 1 dependiendo si tuvo retorno positivo o negativo

data_ret=np.log(data/data.shift(1)).dropna()
data_ret['target']=np.ones((len(data_ret),1))

for i in range(0,len(data_ret)):
    if data_ret.iloc[i,0]<0:
        data_ret.iloc[i,10]=0

# 3.1 Ajuste una regresión Logistica a los datos y calcule:

X = data_ret.iloc[:,1:10].values
y = data_ret.iloc[:,10].values

from sklearn.linear_model import LogisticRegression      
model=LogisticRegression()
model.fit(X,y)

from sklearn import metrics

# 3.1 Accuracy Score

y_pred=model.predict(X)
acc=metrics.accuracy_score(y,y_pred)

# 3.2 Confusion Matrix

cm=metrics.confusion_matrix(y,y_pred)

# 3.3 Precision

prec=metrics.precision_score(y,y_pred)

# 3.4 Recall

recall=metrics.recall_score(y,y_pred)

# 3.5 F1 Score

f1=metrics.f1_score(y,y_pred)

# 3.6 ROC AUC Score

roc_auc=metrics.roc_auc_score(y,y_pred)

# 3.7 Grafique la curva ROC () 

def plot_roc_curve(fpr,tpr,label=None):
    import matplotlib.pyplot as plt
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title("ROC Curve")
    plt.show()

y_score = model.decision_function(X)
(fpr, tpr,thresholds)=metrics.roc_curve(y, y_score)
plot_roc_curve(fpr,tpr)

# 3.8 Calcule la Probabilidad de cada clasificación y compare la regla de clasificación, 
# la clase predicha y la clase real

p=model.predict_proba(X) # versus y_pred versus y

#3.9  Realice la clasificación multiclase (8 clases) utilizando el citerio OnevsRest
# Calcule la probabilidad predicha, su consistencia con la clasificación efectiva y la clase real

data_ret['target']=np.empty((len(data_ret),1))

for i in range(0,len(data_ret)):
    if (data_ret.iloc[i,0]<-0.01):
        data_ret.iloc[i,10]=0
    elif (data_ret.iloc[i,0]>=-0.01)&(data_ret.iloc[i,0]<-0.0075):
        data_ret.iloc[i,10]=1
    elif (data_ret.iloc[i,0]>=-0.0075)&(data_ret.iloc[i,0]<-0.005):
        data_ret.iloc[i,10]=2
    elif (data_ret.iloc[i,0]>=-0.05)&(data_ret.iloc[i,0]<-0.0025):
        data_ret.iloc[i,10]=3
    elif (data_ret.iloc[i,0]>=-0.0025)&(data_ret.iloc[i,0]<0.0):
        data_ret.iloc[i,10]=4
    elif (data_ret.iloc[i,0]>=0.0)&(data_ret.iloc[i,0]<0.0025):
        data_ret.iloc[i,10]=5
    elif (data_ret.iloc[i,0]>=0.0025)&(data_ret.iloc[i,0]<0.005):
        data_ret.iloc[i,10]=6
    elif (data_ret.iloc[i,0]>=0.005)&(data_ret.iloc[i,0]<0.01):
        data_ret.iloc[i,10]=7
    elif (data_ret.iloc[i,0]>=0.01):    
        data_ret.iloc[i,10]=8

X = data_ret.iloc[:,1:10].values
y = data_ret.iloc[:,10].values
        
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

model = OneVsRestClassifier(LogisticRegression())
model.fit(X,y)      

p=model.predict_proba(X)
y_pred=model.predict(X)
        
# Se debe comparar p con y_pred e y

     


