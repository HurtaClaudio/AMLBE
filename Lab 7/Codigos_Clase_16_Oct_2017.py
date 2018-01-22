# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:42:17 2017

@author: Raul
"""

#Entrenamiento de Modelos 

from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#X=2*np.random.rand(100,1) #--------->si quieren generar la data de cero
#y=4+3*X+np.random.randn(100,1) # ---> si quieren generar la data de cero

#%%%
import pandas as pd

location = r'data_gd.xlsm'

X=pd.read_excel(location ,'Sheet1');X=X.values    #  ---> Si quiere tener la misma data de las láminas 
y=pd.read_excel(location ,'Sheet2');y=y.values   # ---> Si quiere tener la misma data de las láminas

#%%

import scipy.stats as s
import numpy as np

X_b=np.c_[np.ones((100,1)),X]
n=float(len(X_b))
k=float(int(X_b.shape[1]))
theta_best=np.dot(np.dot(np.linalg.inv(np.dot(X_b.T,X_b)),X_b.T),y)
y_est=np.dot(X_b,theta_best)
y_prom=y_est.mean()
r2=(np.dot(np.dot(theta_best.T,X_b.T),y)-n*y_prom**2)/(np.dot(y.T,y)-n*y_prom**2)
r2atheil=1-(1-r2)*(n-1)/(n-k)
r2agoldberger=(1-(k/n))*r2
f_test=(r2/(k-1))/((1-r2)/(n - k))
v=np.var(y-y_est)
v_theta=np.linalg.inv(np.dot(X_b.T,X_b))*v
sig_theta=np.sqrt(np.diag(v_theta)).reshape(len(v_theta),1)
t_test=theta_best/sig_theta
te=np.sqrt(v)

ndc=0.05
tc=s.t.ppf(1-ndc,n-k)
fc=s.f.ppf(1-ndc,n-1,n-k)

invtc=s.t.cdf(tc,n-k)
invfc=s.f.cdf(fc,n-1,n-k)


#%%

import statsmodels.formula.api as smf
est = smf.OLS(y,X_b).fit()
est.summary() 

#%%

import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.axis([0,2,0,15])
plt.title('Data Generada')
plt.show()


#df1=pd.DataFrame(X)
#df2=pd.DataFrame(y)
#writer = pd.ExcelWriter('data_gd.xlsx', engine='xlsxwriter')
#df1.to_excel(writer, sheet_name='Sheet1')
#df2.to_excel(writer, sheet_name='Sheet2')
#writer.save()

#%%

X_new=np.array([[0],[2]])
X_new_b=np.c_[np.ones((2,1)),X_new]
y_predict=np.dot(X_new_b,theta_best)

#%%

import matplotlib.pyplot as plt

plt.plot(X_new,y_predict,"r-")
plt.plot(X,y,"b.")
plt.axis([0,2,0,15])
plt.title('Predicciones')
plt.show()

#%%

theta=[]
from sklearn.linear_model import LinearRegression
import sklearn.metrics as m
import numpy as np

lin_reg=LinearRegression()
lin_reg=lin_reg.fit(X,y)
theta.append(lin_reg.intercept_)
theta.append(lin_reg.coef_)
lin_reg.predict(X_new)
y_est=lin_reg.predict(X)
r2=m.r2_score(y,y_est)

#yy = [ float(x) for x in y ]
#yy_est = [ float(x) for x in y_est ]


#f_test=m.fbeta_score(y,y_est,average=None, beta = 0.5)

#%% Regresión Logística

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])
plt.title("Gráfico de la Función Logística")
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())
print(iris.DESCR)

X = iris["data"][:, 3:]  # ancho de pétalo
y = (iris["target"] == 2).astype(np.int)  # 1 si Iris-Virginica, si no 0
        
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="No Iris-Virginica")
plt.text(decision_boundary+0.02, 0.15, "Limite de Decision", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Ancho de Petalo (cm)", fontsize=14)
plt.ylabel("Probabilidad", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.title("Grafico Regresion Logistica")
plt.show()