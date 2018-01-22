#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:23:09 2017

@author: caco
"""
import pandas as pd
import numpy as np

location = r'Data_Clase_PCA.xlsx'

x = pd.read_excel(location)

x_mean = np.mean(x, axis = 0)
x_std = np.std(x, axis=0)
x_norm = (x-x_mean)/x_std

cov = np.cov(x_norm, rowvar=0)


(evals, evects) = np.linalg.eig(cov)
wf = evals/np.sum(evals)

#Paso 4: Generar los "factores" no correlacionados
factors = np.dot(evects.T, x_norm.T).T

#Paso 6: Reconstruccion del sistema con factores reducidos
n_fact = 2

evects_sub= evects[:,0:n_fact]
factors_sub = factors[:,0:n_fact]

proj = pd.DataFrame(np.dot(evects_sub, factors_sub.T).T,index = x.index, columns= x.columns)*x_std +x_mean

drift = proj - x
desc_driftsas = drift.describe()