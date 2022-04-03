# -*- coding: utf-8 -*-
"""EDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VVFNGFbSlSe8s-wVbsraPOZekaZusnCT

**EDA**
"""

fra20.printSchema()

fra20.summary().show()

from pyspark.sql.functions import isnull, when, count, col

fra20.select([count(when(isnull(c), c)).alias(c) for c in fra20.columns]).show()

from pyspark.mllib.stat import Statistics

# select variables to check correlation

df_features = fra20.select('back_legroom','city_fuel_economy','daysonmarket','dealer_zip','engine_displacement','front_legroom','fuel_tank_volume','height','highway_fuel_economy','horsepower','length','maximum_seating','mileage',
                           'owner_count','seller_rating','torque','wheelbase','width','transmission_id','wheelsystem_id','fueltype_id ','bodytype_id ','enginetype_id','is_newIndex','franchise_dealerIndex')

# create RDD table for correlation calculation
rdd_table = df_features.rdd.map(lambda row: row[0:])

# get the correlation matrix
corr_mat=Statistics.corr(rdd_table, method="pearson")
print(corr_mat)
fra20.corr(col1='highway_fuel_economy', col2='city_fuel_economy', method='pearson')

# Visualization

fig=plt.subplots(figsize=(100, 100))

for i, feature in enumerate(['fuel_tank_volume', 'height', 'length', 'mileage', 'owner_count','torque', 'width', 'maximum_seating', 'fueltype_id ','bodytype_id ', 'enginetype_id', 'wheelsystem_id', 'price']):
    plt.subplot(20, 20, i+1)
    plt.subplots_adjust(hspace = 2.0)
    sns.distplot(pd_data[feature])
    plt.tight_layout()


def histogram(fra20, col, bins=10, xname=None, yname=None):
    
    '''
    This function makes a histogram from spark dataframe named 
    df for column name col. 
    '''
    
    # Calculating histogram in Spark 
    vals = fra20.select(col).rdd.flatMap(lambda x: x).histogram(bins)
    
    # Preprocessing histogram points and locations 
    width = vals[0][1] - vals[0][0]
    loc = [vals[0][0] + (i+1) * width for i in range(len(vals[1]))]
     
    # Making a bar plot 
    plt.bar(loc, vals[1], width=width)
    plt.xlabel(col)
    plt.ylabel(yname)
    plt.show()


histogram(fra20, 'mileage', bins=15, yname='enginetype_id')


histogram(fra20, 'city_fuel_economy', bins=15, yname='mileage')


histogram(fra20, 'width', bins=15, yname='height')


histogram(fra20, 'width', bins=15, yname='torque')


histogram(fra20, 'engine_displacement', bins=15, yname='enginetype_id')


histogram(fra20, 'enginetype_id', bins=15, yname='bodytype_id ')


histogram(fra20, 'bodytype_id ', bins=15, yname='price')

histogram(fra17, 'enginetype_id', bins=15, yname='price')


<<<<<<< HEAD


plt.figure(figsize=(15,5))
bodytype_id = pd["bodytype_id "]
colors = ['#78C850', '#F08030', '#6890F0','#F8D030', '#F85888', '#705898', '#98D8D8']
sns.boxplot(pd['width'], y=bodytype_id , palette=colors)
plt.show()


plt.figure(figsize=(15,5))
bodytype_id = pd["bodytype_id "]
colors = ['#78C850', '#F08030', '#6890F0','#F8D030', '#F85888', '#705898', '#98D8D8']
sns.boxplot(pd['width'], y=bodytype_id , palette=colors)
plt.show()


plt.figure(figsize=(15,5))
bodytype_id = pd["enginetype_id"]
colors = ['#78C850', '#F08030', '#6890F0','#F8D030', '#F85888', '#705898', '#98D8D8']
sns.boxplot(pd['width'], y=bodytype_id , palette=colors)
plt.show()


sns.pairplot(pd[['maximum_seating', 'mileage', 'owner_count','engine_displacement','torque','transmission_id', 'wheelsystem_id']], diag_kind = 'kde')
#sns.pairplot(data=pd, x_vars=['maximum_seating', 'mileage', 'owner_count'], kind = 'kde', y_vars='width')
plt.show()
=======

>>>>>>> 9a67c39101b7670b813da351c05eda8d8f276d40
