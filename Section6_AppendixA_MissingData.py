# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 04:05:03 2017

@author: volak
"""

import numpy as np   #Mathematics library
import matplotlib.pyplot as plt # for plotting
import pandas as pd  #manage datasets
import seaborn as sea
#Importing Datasets
feature_data = pd.read_csv('properties_2016.csv')
response_data= pd.read_csv('train_2016.csv')




#########################MergeData  
df = pd.merge(response_data,feature_data,on="parcelid",how="left") 



print(df.isnull().sum())





#Convert date to useable categorical variable
df['transactiondate'] = pd.to_datetime(df.transactiondate)
df['month']=df.transactiondate.dt.month
df['day']=df.transactiondate.dt.weekday_name
df=df.drop('transactiondate',axis=1)


#Anaylzing each column
print(df['hashottuborspa'].nunique())
print(df['hashottuborspa'].unique())
print('\n')
print(df['propertycountylandusecode'].nunique())
print(df['propertycountylandusecode'].unique())
print('\n')
print(df['propertyzoningdesc'].nunique())
print(df['propertyzoningdesc'].unique())
print('\n')
print(df['fireplaceflag'].nunique())
print(df['fireplaceflag'].unique())
print('\n')
print(df['taxdelinquencyflag'].nunique())
print(df['taxdelinquencyflag'].unique()) 
print('\n') 


#replace flags
df['hashottuborspa'] = df['hashottuborspa'].fillna(0)
df['fireplaceflag'] = df['fireplaceflag'].fillna(0)
df['taxdelinquencyflag'] = df['taxdelinquencyflag'].fillna(0)

#---  replace the string 'True' and 'Y' with value '1' ---
df['hashottuborspa'].replace( 'True', 1, inplace=True)
df['fireplaceflag'].replace( 'True',1, inplace=True)
df['taxdelinquencyflag'].replace( 'Y',1, inplace=True)


#--- Since there is only ONE missing value in this column we will replace it manually ---
df["propertycountylandusecode"].fillna('023A', inplace =True)
print(df['propertycountylandusecode'].isnull().sum())


df["propertyzoningdesc"].fillna('UNIQUE', inplace =True)
print(df['propertyzoningdesc'].isnull().sum())

""" 
5.3.2.a Columns regionidcity, regionidneighborhood and regionidzip
Here I am imputing missing values by randomly assigning values already 
present in the respective columns. (If you have a better way to impute such 
values do mention it in the comments section!!)
"""
'''
cols = ['regionidcity', 'regionidneighborhood', 'regionidzip']
print(df['regionidcity'].isnull().sum())
print(df['regionidneighborhood'].isnull().sum())
print(df['regionidzip'].isnull().sum())

df["regionidcity"].fillna(lambda x: random.choice(df[df["regionidcity"] != np.nan]), inplace =True)
df["regionidneighborhood"].fillna(lambda x: random.choice(df[df["regionidneighborhood"] != np.nan]), inplace =True)
df["regionidzip"].fillna(lambda x: random.choice(df[df["regionidzip"] != np.nan]), inplace =True)
df["regionidcounty"].fillna(lambda x: random.choice(df[df["regionidcounty"] != np.nan]), inplace =True)
'''

df['regionidcity'] = df['regionidcity'].fillna(0)
df['regionidneighborhood'] = df['regionidneighborhood'].fillna(0)
df['regionidzip'] = df['regionidzip'].fillna(0)
df['regionidcounty'] = df['regionidcounty'].fillna(0)

#Column unitcnt - Here we will replace missing values with the mostly occuring variable
print(df['unitcnt'].unique())
print(df['unitcnt'].value_counts())
sea.countplot(x = 'unitcnt', data = df)

#--- Replace the missing values with the maximum occurences ---
df['unitcnt'] = df['unitcnt'].fillna(df['unitcnt'].mode()[0])

#--- cross check for missing values ---
print(df['unitcnt'].isnull().sum())

print(df.isnull().sum())

##year built
df['yearbuilt'] = df['yearbuilt'].fillna(2016)

#The remaining columns can be safely assigned value '0', beacuse they all signify a presence of a particular ID or count.
#--- list of columns of type 'float' having missing values
#--- float_nan_col 

#--- list of columns of type 'float' after imputing missing values ---

### Column censustractandblock


'''
df=df.drop('latitude',axis=1)
df=df.drop('longitude',axis=1)
'''

df["propertyzoningdesc"].fillna('UNIQUE', inplace =True)
df["propertyzoningdesc"].fillna(value=0, inplace =True)
#df["regionidcity"].fillna(lambda x: random.choice(df[df["regionidcity"] != np.nan]), inplace =True)
df['fireplaceflag'] = df['fireplaceflag'].fillna(0)

df['airconditioningtypeid'] = df['airconditioningtypeid'].fillna(0)
df['architecturalstyletypeid'] = df['architecturalstyletypeid'].fillna(0)
df['fireplacecnt'] = df['fireplacecnt'].fillna(0)
df['garagecarcnt'] = df['garagecarcnt'].fillna(0)
df['garagetotalsqft'] = df['garagetotalsqft'].fillna(0)
df['heatingorsystemtypeid'] = df['heatingorsystemtypeid'].fillna(0)
df['lotsizesquarefeet'] = df['lotsizesquarefeet'].fillna(0)
df['poolcnt'] = df['poolcnt'].fillna(0)
df['pooltypeid2'] = df['pooltypeid2'].fillna(0)
df['pooltypeid7'] = df['pooltypeid7'].fillna(0)
df['storytypeid'] = df['storytypeid'].fillna(0)
df['threequarterbathnbr'] = df['threequarterbathnbr'].fillna(0)
df['typeconstructiontypeid'] = df['typeconstructiontypeid'].fillna(0)
df['taxdelinquencyyear'] = df['taxdelinquencyyear'].fillna(0)
df['pooltypeid10'] = df['pooltypeid10'].fillna(0)




df['basementsqft'] = df['basementsqft'].fillna(df['basementsqft'].mode()[0])
df['bathroomcnt'] = df['bathroomcnt'].fillna(df['bathroomcnt'].mode()[0])
df['bedroomcnt'] = df['bedroomcnt'].fillna(df['bedroomcnt'].mode()[0])
df['buildingclasstypeid'] = df['buildingclasstypeid'].fillna(df['buildingclasstypeid'].mode()[0])
df['buildingqualitytypeid'] = df['buildingqualitytypeid'].fillna(df['buildingqualitytypeid'].mode()[0])
df['calculatedbathnbr'] = df['calculatedbathnbr'].fillna(df['calculatedbathnbr'].mode()[0])
df['decktypeid'] = df['decktypeid'].fillna(df['decktypeid'].mode()[0])
df['fips'] = df['fips'].fillna(df['fips'].mode()[0])
df['fullbathcnt'] = df['fullbathcnt'].fillna(df['fullbathcnt'].mode()[0])
df['latitude'] = df['latitude'].fillna(df['latitude'].mode()[0])
df['longitude'] = df['longitude'].fillna(df['longitude'].mode()[0])
df['propertylandusetypeid'] = df['propertylandusetypeid'].fillna(df['propertylandusetypeid'].mode()[0])
df['roomcnt'] = df['roomcnt'].fillna(df['roomcnt'].mode()[0])
df['structuretaxvaluedollarcnt'] = df['structuretaxvaluedollarcnt'].fillna(df['structuretaxvaluedollarcnt'].mode()[0])
df['landtaxvaluedollarcnt'] = df['landtaxvaluedollarcnt'].fillna(df['landtaxvaluedollarcnt'].mode()[0])
df['taxvaluedollarcnt'] = df['taxvaluedollarcnt'].fillna(df['taxvaluedollarcnt'].mode()[0])

#drop either empty or zero value
df=df.drop('rawcensustractandblock',axis=1)
df=df.drop('censustractandblock',axis=1)
df=df.drop('assessmentyear',axis=1)

df=df.drop('finishedfloor1squarefeet',axis=1)

df=df.drop('finishedsquarefeet13',axis=1)
df=df.drop('finishedsquarefeet15',axis=1)
df=df.drop('finishedsquarefeet50',axis=1)

df=df.drop('finishedsquarefeet6',axis=1)
df=df.drop('poolsizesum',axis=1)
df=df.drop('yardbuildingsqft17',axis=1)
df=df.drop('yardbuildingsqft26',axis=1)


mean_values = df.mean(axis=0)
df = df.fillna(mean_values, inplace=True)

median_values = df['numberofstories'].median(axis=0)
df['numberofstories'] = df['numberofstories'].fillna(median_values, inplace=True)
median_values = df['calculatedfinishedsquarefeet'].median(axis=0)
df['calculatedfinishedsquarefeet'] = df['calculatedfinishedsquarefeet'].fillna(median_values, inplace=True)
median_values = df['finishedsquarefeet12'].median(axis=0)
df['finishedsquarefeet12'] = df['finishedsquarefeet12'].fillna(median_values, inplace=True)


pd.isnull(df).values.any()
print(df.isnull().sum())


#--- how old is the house? ---
df['house_age'] = 2017 - df['yearbuilt']
#--- how many rooms are there? ---  
df['tot_rooms'] = df['bathroomcnt'] + df['bedroomcnt']


df.to_csv('FinishMissing.csv')
