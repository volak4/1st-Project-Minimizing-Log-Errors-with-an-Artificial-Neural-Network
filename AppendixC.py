# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 05:17:01 2017

@author: volak
"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import scipy.stats as stats
import sklearn

feature_data = pd.read_csv('properties_2016.csv')
response_data= pd.read_csv('train_2016.csv')

fd = feature_data
rd = response_data

merged = pd.merge(response_data,feature_data,on="parcelid",how="left") 
df = merged


#Convert date to useable categorical variable
df['transactiondate'] = pd.to_datetime(df.transactiondate)
df['month']=df.transactiondate.dt.month
df['day']=df.transactiondate.dt.weekday_name
df=df.drop('transactiondate',axis=1)


# Continous = Logerror vs 2 features Linear Regresson
sea.regplot(x='basementsqft', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='bathroomcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='bedroomcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='calculatedbathnbr', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='threequarterbathnbr', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedfloor1squarefeet', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='calculatedfinishedsquarefeet', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedsquarefeet6', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedsquarefeet12', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedsquarefeet13', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedsquarefeet15', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedsquarefeet50', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='fireplacecnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='fullbathcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='garagecarcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='garagetotalsqft', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='latitude', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='longitude', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='lotsizesquarefeet', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='numberofstories', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='poolcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='poolsizesum', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='roomcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='unitcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='yardbuildingsqft17', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='yardbuildingsqft26', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='yearbuilt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='taxvaluedollarcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='structuretaxvaluedollarcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='landtaxvaluedollarcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='taxamount', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='assessmentyear', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='taxdelinquencyflag', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='taxdelinquencyyear', y="logerror", data=df,color="#04cbdb")
            
            
            #Impute Median all at once
median_values = df.median(axis=0)
df = df.fillna(median_values, inplace=True)


# Continous = Logerror vs 2 features Linear Regresson with Imputed Median Values
sea.regplot(x='basementsqft', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='bathroomcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='bedroomcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='calculatedbathnbr', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='threequarterbathnbr', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedfloor1squarefeet', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='calculatedfinishedsquarefeet', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedsquarefeet6', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedsquarefeet12', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedsquarefeet13', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedsquarefeet15', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='finishedsquarefeet50', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='fireplacecnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='fullbathcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='garagecarcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='garagetotalsqft', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='latitude', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='longitude', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='lotsizesquarefeet', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='numberofstories', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='poolcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='poolsizesum', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='roomcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='unitcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='yardbuildingsqft17', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='yardbuildingsqft26', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='yearbuilt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='taxvaluedollarcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='structuretaxvaluedollarcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='landtaxvaluedollarcnt', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='taxamount', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='assessmentyear', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='taxdelinquencyflag', y="logerror", data=df,color="#04cbdb")
sea.regplot(x='taxdelinquencyyear', y="logerror", data=df,color="#04cbdb")