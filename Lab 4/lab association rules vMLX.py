# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:05:49 2017

@author: David Diaz
"""

import pandas as pd #carga librerias
from mlxtend.frequent_patterns import apriori #carga libreria de patrones frecuentes apriori
from mlxtend.frequent_patterns import association_rules #carga libreria reglas de asociación

#defino y leo archivo en formato tabular
Location = r'Baskets_Tabular.csv'
df = pd.read_csv(Location)

## comienzo prepreoceso de datos, la idea es llevar toda la data a formato columnas "dummy" 1 o 0

#reeamplazo T y F por 1 o 0 para columnas con las categorias
df.iloc[:,7:18]=df.iloc[:,7:18].replace(["T","F"],[1,0])

# creo intervalos o categorías para las variables numéricas o continuas
# estoy creando tres intervalos, cada intervalo contendra la misma cantidad de datos
# uso comando qcut (quantile cut)
df['age_rango']=pd.qcut(df['age'],3)
df['age_cat']=pd.qcut(df['age'],3,labels=["baja","media","alta"])

df['value_rango']=pd.qcut(df['value'],3)
df['value_cat']=pd.qcut(df['value'],3,labels=["bajo","medio","alto"])
    
df['income_rango']=pd.qcut(df['income'],3)
df['income_cat']=pd.qcut(df['income'],3,labels=["bajo","medio","alto"])

#para las variables que ya son categóricas, las transformo a dummies
#uso comando get.dummies
df_dummies=pd.get_dummies(df[['sex','homeown','age_cat','value_cat','income_cat','pmethod']])

#ahora que ya esta preprocesaso, sólo me falta concatenar las columnas preprocesadas
basket_sets=pd.concat([df.iloc[:,7:18],df_dummies],axis=1)

#basquet_sets es la data lista para ser analizada

#tomaré una muestra para crear reglas

basket_sets1=basket_sets.iloc[0:995,:]

basket_sets2=basket_sets.iloc[995:,:]

#calculo la frecuencia (soporte) de cada ítem por separado y en conjunto con los otros, dado un soporte mínimo deseado
frequent_itemsets = apriori(basket_sets1, min_support=0.07, use_colnames=True)

#una vez creada la matriz de ítems frecuentes, puedo crear las reglas de asociación
#también puedo filtar para un mínimo deseado, en este caso, métrica lift mínimo 2
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=2)

#ordeno por lift de manera descendente
rules=rules.sort_values(by='lift',ascending=False)

#calculo largo antecedentes de regla
rules["antecedant_len"] = rules["antecedants"].apply(lambda x: len(x))

#calculo largo consecuentes de regla
rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))

#calculo largo consecuentes de regla
rules["rule_lenght"] = rules["consequents_len"]+rules["antecedant_len"]

#aplico algunos filtros de ejemplo

## regla tiene que tener más de 2 antecedentes, confianza mayor a 75% y lift> 2.2
rules_filtro1=rules[
        (rules['antecedant_len'] >= 2) &
        (rules['confidence'] > 0.75) &
        (rules['lift'] > 2.2)]

## regla tiene que tener como consecuente una cerveza, confianza mayor a 10%, lift> 2.2 y un sólo ítem en el consecuente (cerveza)
rules_filtro2=rules[
        (rules['consequents'].astype('str').str.count('beer') ==1) &
        (rules['confidence'] > 0.1) &
        (rules['lift'] > 2.2) &
        (rules["consequents_len"] ==1)]

#genero un mini sistema recomendador
## para canasta en la muestra basket2
for (i, row) in basket_sets2.iterrows():
    canasta=set()    
    a=row.index
    colu=0
    for col in row:
        if col==1:
            canasta.add(a[colu])
        colu=colu+1
    
    ## chequeo que antecedentes cumple, recomiendo el consecuente
    regla=0
    for rul in rules.iterrows():
        ant=rules['antecedants'].iloc[regla]
        cons=rules['consequents'].iloc[regla]
        conf=rules['confidence'].iloc[regla]
        recom=canasta.issuperset(ant) #s.issuperset(t)	s >= t	test whether every element in t is in s
        regla=regla+1
        if recom==True: print(i,ant,"->",cons,conf)
        