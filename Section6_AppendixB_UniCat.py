# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 04:46:10 2017

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



 # Univariate Analysis Features
#Distribution Categorical 
plot = df['airconditioningtypeid'].hist()
plot  = df['architecturalstyletypeid'].hist()
plot  = df['buildingqualitytypeid'].hist()
plot  = df['buildingclasstypeid'].hist()
plot  = df['decktypeid'].hist()
plot  = df['fips'].hist()
plot  = df['fireplaceflag'].hist()
plot  = df['hashottuborspa'].hist()
plot  = df['heatingorsystemtypeid'].hist()
plot = df['parcelid'].hist()
plot = df['pooltypeid10'].hist()
plot  = df['pooltypeid2'].hist()
plot  = df['pooltypeid7'].hist()
plot  = df['propertycountylandusecode'].hist()
plot  = df['propertylandusetypeid'].hist()
plot  = df['propertyzoningdesc'].hist()
plot  = df['rawcensustractandblock'].hist()
plot  = df['censustractandblock'].hist()
plot  = df['regionidcounty'].hist()
plot  = df['regionidcity'].hist()
plot  = df['regionidzip'].hist()
plot  = df['regionidneighborhood'].hist()
plot  = df['storytypeid'].hist()
plot  = df['typeconstructiontypeid'].hist()





#Impute Median all at once
median_values = df.median(axis=0)
df = df.fillna(median_values, inplace=True)


               # Univariate Analysis Features
#Distribution Categorical 
sea.barplot( x = df['airconditioningtypeid'].unique(), y = df['airconditioningtypeid'].value_counts(), data = df)
sea.barplot( x = df['architecturalstyletypeid'].unique(), y = df['architecturalstyletypeid'].value_counts(), data = df)
sea.barplot( x = df['buildingqualitytypeid'].unique(), y = df['buildingqualitytypeid'].value_counts(), data = df)
sea.barplot( x = df['buildingclasstypeid'].unique(), y = df['buildingclasstypeid'].value_counts(), data = df)
sea.barplot( x = df['decktypeid'].unique(), y = df['decktypeid'].value_counts(), data = df)
sea.barplot( x = df['fips'].unique(), y = df['fips'].value_counts(), data = df)
sea.barplot( x = df['fireplaceflag'].unique(), y = df['fireplaceflag'].value_counts(), data = df)
sea.barplot( x = df['hashottuborspa'].unique(), y = df['hashottuborspa'].value_counts(), data = df)
sea.barplot( x = df['heatingorsystemtypeid'].unique(), y = df['heatingorsystemtypeid'].value_counts(), data = df)
sea.barplot( x = df['parcelid'].unique(), y = df['parcelid'].value_counts(), data = df)
sea.barplot( x = df['pooltypeid10'].unique(), y = df['pooltypeid10'].value_counts(), data = df)
sea.barplot( x = df['pooltypeid2'].unique(), y = df['pooltypeid2'].value_counts(), data = df)
sea.barplot( x = df['pooltypeid7'].unique(), y = df['pooltypeid7'].value_counts(), data = df)
sea.barplot( x = df['propertycountylandusecode'].unique(), y = df['propertycountylandusecode'].value_counts(), data = df)
sea.barplot( x = df['propertylandusetypeid'].unique(), y = df['propertylandusetypeid'].value_counts(), data = df)
sea.barplot( x = df['propertyzoningdesc'].unique(), y = df['propertyzoningdesc'].value_counts(), data = df)
sea.barplot( x = df['rawcensustractandblock'].unique(), y = df['rawcensustractandblock'].value_counts(), data = df)
sea.barplot( x = df['censustractandblock'].unique(), y = df['censustractandblock'].value_counts(), data = df)
sea.barplot( x = df['regionidcounty'].unique(), y = df['regionidcounty'].value_counts(), data = df)
sea.barplot( x = df['regionidcity'].unique(), y = df['regionidcity'].value_counts(), data = df)
sea.barplot( x = df['regionidzip'].unique(), y = df['regionidzip'].value_counts(), data = df)
sea.barplot( x = df['regionidneighborhood'].unique(), y = df['regionidneighborhood'].value_counts(), data = df)
sea.barplot( x = df['storytypeid'].unique(), y = df['storytypeid'].value_counts(), data = df)
sea.barplot( x = df['typeconstructiontypeid'].unique(), y = df['typeconstructiontypeid'].value_counts(), data = df)
