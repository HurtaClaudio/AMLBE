# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:57:59 2017

@author: David Diaz
"""

import pandas as pd #panda dataframes

Location = r'TablasImpago.xlsx'

#importo tabla 1 - Muestra 1

Tabla1 = pd.read_excel(Location,sheet_name="Tabla 1")

##importo tabla 2 - Muestra 2
Tabla2 = pd.read_excel(Location,sheet_name="Tabla 2")


#primera operación Append - coloco una tabla a continuación de la otra, todas las columnas coinciden en ambas tablas

Tabla_A=Tabla1
Tabla_A=Tabla_A.append(Tabla2)

#manera alternativa usando concat
#vertical
Tabla_A_alt_V=pd.concat([Tabla1,Tabla2],axis=0)

#horizontal, equivalente a BuscarV usando index como llave
Tabla_A_alt_H=pd.concat([Tabla1,Tabla2],axis=1)

#ignorando indice
Tabla_A_alt_H2=pd.concat([Tabla1,Tabla2],axis=1, ignore_index=True)

##importo tabla 3
Tabla3 = pd.read_excel(Location,sheet_name="Tabla 3")

## inner join, usando indice como llave, equivalente a BuscarV Merge

Tabla_B=pd.merge(Tabla_A,Tabla3)

## inner join, usando ID2 como llave, equivalente a BuscarV Merge

Tabla_B2=pd.merge(Tabla_A,Tabla3,on='ID2')

## inner join, usando ID1, ID2 como llave (llave compuesta), equivalente a BuscarV Merge

Tabla_B3=pd.merge(Tabla_A,Tabla3,on=['ID1','ID2'])

## creo Tabla C, tomo una muestra de la Tabla A

Tabla_C=Tabla_A.sample(frac=0.7,replace=False)

## creo Tabla D, tomo una muestra de la Tabla B

Tabla_D=Tabla_B.sample(frac=0.2,replace=False)


## inner Join, usando ID2

Tabla_E=pd.merge(Tabla_C,Tabla_D, on='ID2')

## left Join, Tabla C, Tabla D, ID2

Tabla_F=pd.merge(Tabla_C,Tabla_D, on='ID2',how='left')

## outer Join, usando ID
Tabla_G=pd.merge(Tabla_C,Tabla_D, on='ID2',how='outer')


## 
#exporting to csv
#in case no header needed header=False, the same for index
Tabla_G.to_csv('Tabla_G.csv')







