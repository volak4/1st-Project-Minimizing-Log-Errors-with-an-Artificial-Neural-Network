# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 04:40:16 2017

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




#### View the unique values of each feature
#Scan to see how the unique values look like for each feature
print ('Unique values A/C', zillow_df.airconditioningtypeid.unique())
print ('Unique valuesarchitecturalstyletypeid',zillow_df.architecturalstyletypeid.unique())
print ('Unique valuesbathroomcnt',zillow_df.bathroomcnt.unique())
print ('Unique valuesbedroomcnt',zillow_df.bedroomcnt.unique())
print ('Unique valuesbuildingclasstypeid',zillow_df.buildingclasstypeid.unique())
print ('Unique valuesbuildingqualitytypeid',zillow_df.buildingqualitytypeid.unique())
print ('Unique valuescalculatedbathnbr',zillow_df.calculatedbathnbr.unique())
print ('Unique valuesdecktypeid',zillow_df.decktypeid.unique())
print ('Unique valuesfinishedfloor1squarefeet',zillow_df.finishedfloor1squarefeet.unique())
print ('Unique valuesfinishedsquarefeet12',zillow_df.finishedsquarefeet12.unique())
print ('Unique valuesfinishedsquarefeet13',zillow_df.finishedsquarefeet13.unique())
print ('Unique valuesfinishedsquarefeet15',zillow_df.finishedsquarefeet15.unique())
print ('Unique valuesfinishedsquarefeet50',zillow_df.finishedsquarefeet50.unique())
print ('Unique valuesfinishedsquarefeet6',zillow_df.finishedsquarefeet6.unique())
print ('Unique valuesfips',zillow_df.fips.unique())
print ('Unique valuesfireplacecnt',zillow_df.fireplacecnt.unique())
print ('Unique valuesfullbathcnt',zillow_df.fullbathcnt.unique())
print ('Unique valuesgaragecarcnt',zillow_df.garagecarcnt.unique())
print ('Unique valuesgaragetotalsqft',zillow_df.garagetotalsqft.unique())
print ('Unique valueshashottuborspa',zillow_df.hashottuborspa.unique())
print ('Unique valuesheatingorsystemtypeid',zillow_df.heatingorsystemtypeid.unique())
print ('Unique valuespoolcnt',zillow_df.poolcnt.unique())
print ('Unique valuespooltypeid7',zillow_df.pooltypeid7.unique())
print ('Unique valuespropertycountylandusecode',zillow_df.propertycountylandusecode.unique())
print ('Unique valuespropertylandusetypeid',zillow_df.propertylandusetypeid.unique())
print ('Unique valuespropertyzoningdesc',zillow_df.propertyzoningdesc.unique())
print ('Unique valuesrawcensustractandblock',zillow_df.rawcensustractandblock.unique())
print ('Unique valuesregionidcity',zillow_df.regionidcity.unique())
print ('Unique valuesregionidcounty',zillow_df.regionidcounty.unique())
print ('Unique valuesregionidneighborhood',zillow_df.regionidneighborhood.unique())
print ('Unique valuesregionidzip',zillow_df.regionidzip.unique())
print ('Unique valuesroomcnt',zillow_df.roomcnt.unique())
print ('Unique valuesstorytypeid',zillow_df.storytypeid.unique())
print ('Unique valuesthreequarterbathnbr',zillow_df.threequarterbathnbr.unique())
print ('Unique valuestypeconstructiontypeid',zillow_df.typeconstructiontypeid.unique())
print ('Unique valuesunitcnt',zillow_df.unitcnt.unique())
print ('Unique valuesyardbuildingsqft17',zillow_df.yardbuildingsqft17.unique())
print ('Unique valuesyardbuildingsqft26',zillow_df.yardbuildingsqft26.unique())
print ('Unique valuesnumberofstories',zillow_df.numberofstories.unique())
print ('Unique valuesfireplaceflag',zillow_df.fireplaceflag.unique())
print ('Unique valuesstructuretaxvaluedollarcnt',zillow_df.structuretaxvaluedollarcnt.unique())
print ('Unique valuestaxvaluedollarcnt',zillow_df.taxvaluedollarcnt.unique())
print ('Unique valuesassessmentyear',zillow_df.assessmentyear.unique())
print ('Unique valueslandtaxvaluedollarcnt',zillow_df.landtaxvaluedollarcnt.unique())
print ('Unique valuestaxamount',zillow_df.taxamount.unique())
print ('Unique valuestaxdelinquencyflag',zillow_df.taxdelinquencyflag.unique())
print ('Unique valuestaxdelinquencyyear',zillow_df.taxdelinquencyyear.unique())
print ('Unique valuescensustractandblock',zillow_df.censustractandblock.unique())



             # Univariate Analysis Response - Logerror histogram
sea.distplot(df['logerror']);                  #histogram
print("Skewness: %f" % df['logerror'].skew())
print("Kurtosis: %f" % df['logerror'].kurt())
res = stats.probplot(df['logerror'], plot=plt) #Q-Q plot

#Distribution Continous 
plot = df['basementsqft'].hist(bins=60,color="#4edde8")
plot = df['bathroomcnt'].hist(bins=60,color="#4edde8")
plot = df['bedroomcnt'].hist(bins=60,color="#4edde8")
plot = df['calculatedbathnbr'].hist(bins=60,color="#4edde8")
plot = df['threequarterbathnbr'].hist(bins=60,color="#4edde8")
plot = df['finishedfloor1squarefeet'].hist(bins=60,color="#4edde8")
plot = df['calculatedfinishedsquarefeet'].hist(bins=60,color="#4edde8")
plot = df['finishedsquarefeet6'].hist(bins=60,color="#4edde8")
plot = df['finishedsquarefeet12'].hist(bins=60,color="#4edde8")
plot = df['finishedsquarefeet13'].hist(bins=60,color="#4edde8")
plot = df['finishedsquarefeet15'].hist(bins=60,color="#4edde8")
plot = df['finishedsquarefeet50'].hist(bins=60,color="#4edde8")
plot = df['fireplacecnt'].hist(bins=60,color="#4edde8")
plot = df['fullbathcnt'].hist(bins=60,color="#4edde8")
plot = df['garagecarcnt'].hist(bins=60,color="#4edde8")
plot = df['garagetotalsqft'].hist(bins=60,color="#4edde8")
plot = df['latitude'].hist(bins=60,color="#4edde8")
plot = df['longitude'].hist(bins=60,color="#4edde8")
plot = df['lotsizesquarefeet'].hist(bins=60,color="#4edde8")
plot = df['numberofstories'].hist(bins=60,color="#4edde8")
plot = df['poolcnt'].hist(bins=60,color="#4edde8")
plot = df['poolsizesum'].hist(bins=60,color="#4edde8")
plot = df['roomcnt'].hist(bins=60,color="#4edde8")
plot = df['unitcnt'].hist(bins=60,color="#4edde8")
plot = df['yardbuildingsqft17'].hist(bins=60,color="#4edde8")
plot = df['yardbuildingsqft26'].hist(bins=60,color="#4edde8")
plot = df['yearbuilt'].hist(bins=60,color="#4edde8")
plot = df['taxvaluedollarcnt'].hist(bins=60,color="#4edde8")
plot = df['structuretaxvaluedollarcnt'].hist(bins=60,color="#4edde8")
plot = df['landtaxvaluedollarcnt'].hist(bins=60,color="#4edde8")
plot = df['taxamount'].hist(bins=60,color="#4edde8")
plot = df['assessmentyear'].hist(bins=60,color="#4edde8")
plot = df['taxdelinquencyflag'].hist(bins=60,color="#4edde8")
plot = df['taxdelinquencyyear'].hist(bins=60,color="#4edde8")

         
         #Impute Median all at once
median_values = df.median(axis=0)
df = df.fillna(median_values, inplace=True)

# Univariate Analysis Response - Logerror histogram
sea.distplot(df['logerror']);                  #histogram
print("Skewness: %f" % df['logerror'].skew())
print("Kurtosis: %f" % df['logerror'].kurt())
res = stats.probplot(df['logerror'], plot=plt) #Q-Q plot

#Distribution Continous 
plot = df['basementsqft'].hist(bins=60,color="#4edde8")
plot = df['bathroomcnt'].hist(bins=60,color="#4edde8")
plot = df['bedroomcnt'].hist(bins=60,color="#4edde8")
plot = df['calculatedbathnbr'].hist(bins=60,color="#4edde8")
plot = df['threequarterbathnbr'].hist(bins=60,color="#4edde8")
plot = df['finishedfloor1squarefeet'].hist(bins=60,color="#4edde8")
plot = df['calculatedfinishedsquarefeet'].hist(bins=60,color="#4edde8")
plot = df['finishedsquarefeet6'].hist(bins=60,color="#4edde8")
plot = df['finishedsquarefeet12'].hist(bins=60,color="#4edde8")
plot = df['finishedsquarefeet13'].hist(bins=60,color="#4edde8")
plot = df['finishedsquarefeet15'].hist(bins=60,color="#4edde8")
plot = df['finishedsquarefeet50'].hist(bins=60,color="#4edde8")
plot = df['fireplacecnt'].hist(bins=60,color="#4edde8")
plot = df['fullbathcnt'].hist(bins=60,color="#4edde8")
plot = df['garagecarcnt'].hist(bins=60,color="#4edde8")
plot = df['garagetotalsqft'].hist(bins=60,color="#4edde8")
plot = df['latitude'].hist(bins=60,color="#4edde8")
plot = df['longitude'].hist(bins=60,color="#4edde8")
plot = df['lotsizesquarefeet'].hist(bins=60,color="#4edde8")
plot = df['numberofstories'].hist(bins=60,color="#4edde8")
plot = df['poolcnt'].hist(bins=60,color="#4edde8")
plot = df['poolsizesum'].hist(bins=60,color="#4edde8")
plot = df['roomcnt'].hist(bins=60,color="#4edde8")
plot = df['unitcnt'].hist(bins=60,color="#4edde8")
plot = df['yardbuildingsqft17'].hist(bins=60,color="#4edde8")
plot = df['yardbuildingsqft26'].hist(bins=60,color="#4edde8")
plot = df['yearbuilt'].hist(bins=60,color="#4edde8")
plot = df['taxvaluedollarcnt'].hist(bins=60,color="#4edde8")
plot = df['structuretaxvaluedollarcnt'].hist(bins=60,color="#4edde8")
plot = df['landtaxvaluedollarcnt'].hist(bins=60,color="#4edde8")
plot = df['taxamount'].hist(bins=60,color="#4edde8")
plot = df['assessmentyear'].hist(bins=60,color="#4edde8")
plot = df['taxdelinquencyflag'].hist(bins=60,color="#4edde8")
plot = df['taxdelinquencyyear'].hist(bins=60,color="#4edde8")

