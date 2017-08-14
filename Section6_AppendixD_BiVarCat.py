# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 05:29:27 2017

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


         ## Part B) Bivarirate
# Categorical = Logerror vs 2 features BOX -PLOT
sea.boxplot(x='airconditioningtypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='architecturalstyletypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='buildingqualitytypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='buildingclasstypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='decktypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='fips', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='fireplaceflag', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='hashottuborspa', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='heatingorsystemtypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='parcelid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='pooltypeid10', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='pooltypeid2', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='pooltypeid7', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='propertycountylandusecode', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='propertylandusetypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='propertyzoningdesc', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='rawcensustractandblock', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='censustractandblock', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='regionidcounty', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='regionidcity', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='regionidzip', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='regionidneighborhood', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='storytypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='typeconstructiontypeid', y="logerror", data=df,color="#4edde8")

            
                        #Impute Median all at once
median_values = df.median(axis=0)
df = df.fillna(median_values, inplace=True)


         ## Part B) Bivarirate
# Categorical = Logerror vs 2 features BOX -PLOT
sea.boxplot(x='airconditioningtypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='architecturalstyletypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='buildingqualitytypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='buildingclasstypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='decktypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='fips', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='fireplaceflag', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='hashottuborspa', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='heatingorsystemtypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='parcelid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='pooltypeid10', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='pooltypeid2', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='pooltypeid7', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='propertycountylandusecode', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='propertylandusetypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='propertyzoningdesc', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='rawcensustractandblock', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='censustractandblock', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='regionidcounty', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='regionidcity', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='regionidzip', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='regionidneighborhood', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='storytypeid', y="logerror", data=df,color="#4edde8")
sea.boxplot(x='typeconstructiontypeid', y="logerror", data=df,color="#4edde8")