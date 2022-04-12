#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pyspark 


# # IMPORT LIBRARIES

# In[2]:


sc.install_pypi_package("matplotlib==3.1.1", "https://pypi.org/simple")
sc.install_pypi_package("pandas==1.2.2")
sc.install_pypi_package("statistics")


# In[3]:


from pyspark.sql import SparkSession
#import warnings
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sns
from pyspark.sql.functions import when, lit, col, substring, substring_index
import pyspark.sql.functions
from pyspark.sql.functions import translate
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler,OneHotEncoder, StringIndexer
import pyspark.sql.functions as F
from pyspark.sql import functions as f
from functools import reduce
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import DecisionTreeRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pyspark.sql.types import Row
from pyspark.ml.linalg import DenseVector
from pyspark.ml.linalg import Vectors


# In[4]:


spark = SparkSession.builder.appName("Spark").getOrCreate()


# # LOAD DATA 

# In[5]:


data=spark.read.parquet("s3://parquetfile07/usedcars__data.parquet/")


# # EXPLORING DATASET 

# In[6]:


type(data)


# In[7]:


data.count()


# In[8]:


len(data.columns)


# In[9]:


data.printSchema()


# In[10]:


data.columns


# # REPARTITIONING

# In[11]:


data.rdd.getNumPartitions()


# In[12]:


data=data.repartition(750)


# In[13]:


data.rdd.getNumPartitions()


# # CLEANING

# width

# In[14]:


wid1=data.withColumn('width', translate('width', 'in', ''))
wid2=wid1.withColumn('width', translate('width', ' ', ''))
wid3=wid2.withColumn('width', translate('width', '.', ' '))
wid4=wid3.withColumn('width', regexp_replace('width', '(\s+\d+\s*)$', ''))
wid5=wid4.withColumn('width', regexp_replace('width', '[^\d{2}]', '0'))
wid6=wid5.withColumn('width', regexp_replace('width', '^(\d{3,1000})', '72'))
wid7=wid6.withColumn('width', regexp_replace('width', '(^0+)', '0'))
wid8=wid7.withColumn('width', regexp_replace('width', '^(\d{1})$', '72'))
wid9=wid8.withColumn('width',col('width').cast("double"))
wid10=wid9.withColumn('width',col('width')).na.fill(73)
# wid10.groupby('width').count().sort('count', ascending=False).collect()


# is_new

# In[15]:


is_n1=wid10.withColumn('is_new', regexp_replace('is_new', '(TRUE)', '1'))
is_n2=is_n1.withColumn('is_new', regexp_replace('is_new', '(FALSE)', '0'))
is_n3=is_n2.withColumn('is_new', regexp_replace('is_new', '[^0-1]', '0'))
is_n4=is_n3.withColumn('is_new', regexp_replace('is_new', '(\d{2,2500})', '0'))
is_n5=is_n4.withColumn('is_new', regexp_replace('is_new', '1', 'TRUE'))
is_n6=is_n5.withColumn('is_new', regexp_replace('is_new', '0', 'FALSE'))
is_n7=is_n6.withColumn('is_new', col('is_new')).na.fill('TRUE')
# is_n7.groupby('is_new').count().sort('count', ascending=False).collect()


# torque

# In[16]:


tor1=is_n7.withColumn('torque', translate('torque', ' ', ''))
tor2 = tor1.withColumn('torque', substring('torque', 1,3))
tor3=tor2.withColumn('torque', regexp_replace('torque', '[^\d{3}]', '0'))
tor4=tor3.withColumn('torque', regexp_replace('torque', '^0[0-4][0-9]', '250'))
tor5=tor4.withColumn('torque', regexp_replace('torque', '^(\d{1,2})$', '250'))
tor6=tor5.withColumn("torque",col("torque").cast("double"))
tor7=tor6.withColumn("torque", col("torque")).na.fill(250)


# daysonmarket

# In[17]:


#day1=tor7.withColumn("daysonmarket",col("daysonmarket").cast("double"))
day2=tor7.withColumn("daysonmarket", col("daysonmarket")).na.fill(76)


# engine_displacement

# In[18]:


engd1=day2.withColumn('engine_displacement', regexp_replace('engine_displacement', '[^\d{4}]', '0')) 
engd2=engd1.withColumn('engine_displacement', regexp_replace('engine_displacement', '^(\d{5,3500})', '2970'))
engd3=engd2.withColumn('engine_displacement', regexp_replace('engine_displacement', '^0[0-9]*$', '2970'))
engd4=engd3.withColumn('engine_displacement', regexp_replace('engine_displacement', '^[0-9]$', '2970'))
engd5=engd4.withColumn("engine_displacement",col("engine_displacement").cast("double"))
engd6=engd5.withColumn("engine_displacement", col("engine_displacement")).na.fill(2970)


# highway_fuel_economy

# In[19]:


hig1=engd6.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '[^\d{3}]', '0')) 
hig2 = hig1.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '(\d{4,3000})', '0')) 
hig3 = hig2.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '^0', '30'))
hig4 = hig3.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '30[0-9]*$', '30'))
hig5=hig4.withColumn("highway_fuel_economy",col("highway_fuel_economy").cast("double"))
hig6=hig5.withColumn("highway_fuel_economy", col("highway_fuel_economy")).na.fill(30)


# wheel_system

# In[20]:


whe1=hig6.withColumn('wheel_system', regexp_replace('wheel_system', 'AWD', '1'))
whe2=whe1.withColumn('wheel_system', regexp_replace('wheel_system', '4WD', '2'))
whe3=whe2.withColumn('wheel_system', regexp_replace('wheel_system', 'FWD', '3'))
whe4=whe3.withColumn('wheel_system', regexp_replace('wheel_system', 'RWD', '4'))
whe5=whe4.withColumn('wheel_system', regexp_replace('wheel_system', '4X2', '5'))
whe6=whe5.withColumn('wheel_system', regexp_replace('wheel_system', '[^1-5]', '0'))
whe7=whe6.withColumn('wheel_system', regexp_replace('wheel_system', '(\d{2,3000})', '3')) 
whe8=whe7.withColumn('wheel_system', regexp_replace('wheel_system', '0', '3'))
whe9=whe8.withColumn('wheel_system', regexp_replace('wheel_system', '1', 'AWD'))
whe10=whe9.withColumn('wheel_system', regexp_replace('wheel_system', '2', '4WD'))
whe11=whe10.withColumn('wheel_system', regexp_replace('wheel_system', '3', 'FWD'))
whe12=whe11.withColumn('wheel_system', regexp_replace('wheel_system', '^4$', 'RWD'))
whe13=whe12.withColumn('wheel_system', regexp_replace('wheel_system', '5', '4X2'))
# whe8=whe7.withColumn("wheel_system",col("wheel_system").cast("double"))
whe14=whe13.withColumn("wheel_system", col("wheel_system")).na.fill('FWD')


# transmission

# In[21]:


tra1=whe14.withColumn('transmission', regexp_replace('transmission', 'CVT', '1')) 
tra2=tra1.withColumn('transmission', regexp_replace('transmission', 'A', '2')) 
tra3=tra2.withColumn('transmission', regexp_replace('transmission', 'M', '3'))
tra4=tra3.withColumn('transmission', regexp_replace('transmission', 'Dual Clutch', '4'))
tra5=tra4.withColumn('transmission', regexp_replace('transmission', '[^1-4]', '0'))
tra6=tra5.withColumn('transmission', regexp_replace('transmission', '(\d{2,2500})', '2'))
tra7=tra6.withColumn('transmission', regexp_replace('transmission', '0', '2'))
# tra7=tra6.withColumn("transmission",col("transmission").cast("double"))
tra8=tra7.withColumn("transmission", col("transmission")).na.fill('2')
tra9=tra8.withColumn('transmission', regexp_replace('transmission', '1', 'CVT')) 
tra10=tra9.withColumn('transmission', regexp_replace('transmission', '2', 'A')) 
tra11=tra10.withColumn('transmission', regexp_replace('transmission', '3', 'M'))
tra12=tra11.withColumn('transmission', regexp_replace('transmission', '4', 'Dual Clutch'))


# back_legroom

# In[22]:


bac1=tra12.withColumn('back_legroom', translate('back_legroom', 'in', ''))
bac2=bac1.withColumn('back_legroom', translate('back_legroom', ' ', ''))
bac3=bac2.withColumn('back_legroom', translate('back_legroom', '.', ' '))
bac4=bac3.withColumn('back_legroom', regexp_replace('back_legroom', '(\s+\d*\s*)$', ''))    #(\s+\d*\s*)$
bac5=bac4.withColumn('back_legroom', regexp_replace('back_legroom', '[^\d{2}]', '0'))
bac6=bac5.withColumn('back_legroom', regexp_replace('back_legroom', '^(\d{3,3000})', '38')) 
bac7=bac6.withColumn('back_legroom', regexp_replace('back_legroom', "(^0+)", "0"))
bac8=bac7.withColumn('back_legroom', regexp_replace('back_legroom', '^0$', '38')) 
bac9=bac8.withColumn('back_legroom', regexp_replace('back_legroom', r'^[0]*', ''))
bac10=bac9.withColumn('back_legroom',when(bac9.back_legroom<=9,38).otherwise(bac9.back_legroom))
bac11=bac10.withColumn('back_legroom',col('back_legroom').cast("double"))
bac12=bac11.withColumn('back_legroom',col('back_legroom')).na.fill(38)


# front_legroom

# In[23]:


fro1=bac12.withColumn('front_legroom', translate('front_legroom', 'in', ''))
fro2=fro1.withColumn('front_legroom', translate('front_legroom', ' ', ''))
fro3=fro2.withColumn('front_legroom', translate('front_legroom', '.', ' '))
fro4=fro3.withColumn('front_legroom', regexp_replace('front_legroom', '(\s+\d*\s*)$', ''))    
fro5=fro4.withColumn('front_legroom', regexp_replace('front_legroom', '[^\d{2}]', '0'))
fro6=fro5.withColumn('front_legroom', regexp_replace('front_legroom', '^(\d{3,3000})', '40')) 
fro7=fro6.withColumn('front_legroom', regexp_replace('front_legroom', "(^0+)", "0"))
fro8=fro7.withColumn('front_legroom', regexp_replace('front_legroom', '^0$', '40')) 
fro9=fro8.withColumn('front_legroom', regexp_replace('front_legroom', r'^[0]*', ''))
fro10=fro9.withColumn('front_legroom',when(fro9.front_legroom<=9,40).otherwise(fro9.front_legroom))
fro11=fro10.withColumn('front_legroom',col('front_legroom').cast("double"))
fro12=fro11.withColumn('front_legroom',col('front_legroom')).na.fill(40)


# height

# In[24]:


hei1=fro12.withColumn('height', translate('height', 'in', ''))
hei2=hei1.withColumn('height', translate('height', ' ', ''))
hei3=hei2.withColumn('height', translate('height', '.', ' '))
hei4=hei3.withColumn('height', regexp_replace('height', '(\s+\d*\s*)$', ''))    
hei5=hei4.withColumn('height', regexp_replace('height', '[^\d{2}]', '0'))
hei6=hei5.withColumn('height', regexp_replace('height', '^(\d{3,3000})', '66')) 
hei7=hei6.withColumn('height', regexp_replace('height', "(^0+)", "0"))
hei8=hei7.withColumn('height', regexp_replace('height', '^0$', '66')) 
hei9=hei8.withColumn('height', regexp_replace('height', r'^[0]*', ''))
hei10=hei9.withColumn('height',when(hei9.height<=9,66).otherwise(hei9.height))
#hei11=hei10.withColumn('height',col('height').cast("double"))
hei12=hei10.withColumn('height',col('height')).na.fill(66)


# length

# In[25]:


len1=hei12.withColumn('length', translate('length', 'in', ''))
len2=len1.withColumn('length', translate('length', ' ', ''))
len3=len2.withColumn('length', translate('length', '.', ' '))
len4=len3.withColumn('length', regexp_replace('length', '(\s+\d*\s*)$', ''))    
len5=len4.withColumn('length', regexp_replace('length', '[^\d{3}]', '0'))
len6=len5.withColumn('length', regexp_replace('length', '^(\d{4,3000})', '231'))
len7=len6.withColumn('length', regexp_replace('length', '^[0-9]{2}$', '231'))
len8=len7.withColumn('length', regexp_replace('length', "(^0+)", "0"))
len9=len8.withColumn('length', regexp_replace('length', '^0$', '231')) 
len10=len9.withColumn('length', regexp_replace('length', r'^[0]*', ''))
len11=len10.withColumn('length',when(len10.length<=9,231).otherwise(len10.length))
len12=len11.withColumn('length',col('length').cast("double"))
len13=len12.withColumn('length',col('length')).na.fill(231)


# fuel_tank_volume

# In[26]:


fue1=len13.withColumn('fuel_tank_volume', translate('fuel_tank_volume', 'gal', ''))
fue2=fue1.withColumn('fuel_tank_volume', translate('fuel_tank_volume', ' ', ''))
fue3=fue2.withColumn('fuel_tank_volume', translate('fuel_tank_volume', '.', ' '))
fue4=fue3.withColumn('fuel_tank_volume', regexp_replace('fuel_tank_volume', '(\s+\d*\s*)$', ''))    
fue5=fue4.withColumn('fuel_tank_volume', regexp_replace('fuel_tank_volume', '[^\d{2}]', '0'))
fue6=fue5.withColumn('fuel_tank_volume', regexp_replace('fuel_tank_volume', '^(\d{3,3000})', '26')) 
fue7=fue6.withColumn('fuel_tank_volume', regexp_replace('fuel_tank_volume', "(^0+)", "0"))
fue8=fue7.withColumn('fuel_tank_volume', regexp_replace('fuel_tank_volume', '^0$', '26')) 
fue9=fue8.withColumn('fuel_tank_volume', regexp_replace('fuel_tank_volume', r'^[0]*', ''))
fue10=fue9.withColumn('fuel_tank_volume',when(fue9.fuel_tank_volume<=9,26).otherwise(fue9.fuel_tank_volume))
fue11=fue10.withColumn('fuel_tank_volume',col('fuel_tank_volume').cast("double"))
fue12=fue11.withColumn('fuel_tank_volume',col('fuel_tank_volume')).na.fill(26)


# body_type

# In[27]:


bod1=fue12.withColumn('body_type', regexp_replace('body_type', 'Coupe', '1')) 
bod2=bod1.withColumn('body_type', regexp_replace('body_type', 'Convertible', '2')) 
bod3=bod2.withColumn('body_type', regexp_replace('body_type', 'Wagon', '3'))
bod4=bod3.withColumn('body_type', regexp_replace('body_type', 'Sedan', '4'))
bod5=bod4.withColumn('body_type', regexp_replace('body_type', 'SUV / Crossover', '5'))
bod6=bod5.withColumn('body_type', regexp_replace('body_type', '[^1-5]', '0'))
bod7= bod6.withColumn('body_type', regexp_replace('body_type', '(\d{2,3000})', '6')) 
#bod8=bod7.withColumn("body_type",bod7["body_type"].cast("double"))
bod8=bod7.withColumn("body_type", col("body_type")).na.fill('6')
bod9=bod8.withColumn('body_type', regexp_replace('body_type', '1', 'Coupe')) 
bod10=bod9.withColumn('body_type', regexp_replace('body_type', '2', 'Convertible')) 
bod11=bod10.withColumn('body_type', regexp_replace('body_type', '3', 'Wagon'))
bod12=bod11.withColumn('body_type', regexp_replace('body_type', '4', 'Sedan'))
bod13=bod12.withColumn('body_type', regexp_replace('body_type', '5', 'SUV / Crossover'))
bod14=bod13.withColumn('body_type', regexp_replace('body_type', '6', 'Other_body'))


# engine_type

# In[28]:


eng1=bod14.withColumn('engine_type', regexp_replace('engine_type', 'I5', '1')) 
eng2=eng1.withColumn('engine_type', regexp_replace('engine_type', 'I5 Diesel', '1'))
eng3=eng2.withColumn('engine_type', regexp_replace('engine_type', 'I5 Biodiesel', '1'))
eng4=eng3.withColumn('engine_type', regexp_replace('engine_type', 'V10', '2')) 
eng5=eng4.withColumn('engine_type', regexp_replace('engine_type', 'V12', '3')) 
eng6=eng5.withColumn('engine_type', regexp_replace('engine_type', 'V8', '4')) 
eng7=eng6.withColumn('engine_type', regexp_replace('engine_type', 'V8 Flex Fuel Vehicle', '4')) 
eng8=eng7.withColumn('engine_type', regexp_replace('engine_type', 'V8 Hybrid', '4')) 
eng9=eng8.withColumn('engine_type', regexp_replace('engine_type', 'V8 Diesel', '4')) 
eng10=eng9.withColumn('engine_type', regexp_replace('engine_type', 'V8 Compressed Natural Gas', '4')) 
eng11=eng10.withColumn('engine_type', regexp_replace('engine_type', 'V6 Diesel', '5')) 
eng12=eng11.withColumn('engine_type', regexp_replace('engine_type', 'V6', '5')) 
eng13=eng12.withColumn('engine_type', regexp_replace('engine_type', 'V6 Flex Fuel Vehicle', '5')) 
eng14=eng13.withColumn('engine_type', regexp_replace('engine_type', 'V6 Hybrid', '5')) 
eng15=eng13.withColumn('engine_type', regexp_replace('engine_type', 'V6 Biodiesel', '5'))
eng16=eng15.withColumn('engine_type', regexp_replace('engine_type', 'I4', '6'))
eng17=eng16.withColumn('engine_type', regexp_replace('engine_type', 'I4 Compressed Natural Gas', '6'))
eng18=eng17.withColumn('engine_type', regexp_replace('engine_type', 'I4 Flex Fuel Vehicle', '6'))
eng19=eng18.withColumn('engine_type', regexp_replace('engine_type', 'I4 Hybrid', '6'))
eng20=eng19.withColumn('engine_type', regexp_replace('engine_type', 'I4 Diesel', '6'))
eng21=eng20.withColumn('engine_type', regexp_replace('engine_type', '[^1-6]', '0'))
eng22=eng21.withColumn('engine_type', regexp_replace('engine_type', '(\d{2,3000})', '7'))
eng23=eng22.withColumn('engine_type', regexp_replace('engine_type', '0', '7'))
#eng24=eng23.withColumn("engine_type",eng23["engine_type"].cast("double"))
eng24=eng23.withColumn("engine_type", col("engine_type")).na.fill('7')
eng25=eng24.withColumn('engine_type', regexp_replace('engine_type', '1', 'I5')) 
eng26=eng25.withColumn('engine_type', regexp_replace('engine_type', '2', 'V10' )) 
eng27=eng26.withColumn('engine_type', regexp_replace('engine_type', '3', 'V12')) 
eng28=eng27.withColumn('engine_type', regexp_replace('engine_type', '^4$', 'V8')) 
eng29=eng28.withColumn('engine_type', regexp_replace('engine_type', '^5$', 'V6')) 
eng30=eng29.withColumn('engine_type', regexp_replace('engine_type', '^6$', 'I4')) 
eng31=eng30.withColumn('engine_type', regexp_replace('engine_type', '7', 'Other_engine')) 


# fuel_type

# In[29]:


fue_t1=eng31.withColumn('fuel_type', regexp_replace('fuel_type', 'Electric', '1')) 
fue_t2=fue_t1.withColumn('fuel_type', regexp_replace('fuel_type', 'Compressed Natural Gas', '2')) 
fue_t3=fue_t2.withColumn('fuel_type', regexp_replace('fuel_type', 'Hybrid', '3')) 
fue_t4=fue_t3.withColumn('fuel_type', regexp_replace('fuel_type', 'Flex Fuel Vehicle', '4')) 
fue_t5=fue_t4.withColumn('fuel_type', regexp_replace('fuel_type', 'Gasoline', '5')) 
fue_t6=fue_t5.withColumn('fuel_type', regexp_replace('fuel_type', 'Biodiesel', '6')) 
fue_t7=fue_t6.withColumn('fuel_type', regexp_replace('fuel_type', 'Diesel', '7')) 
fue_t8=fue_t7.withColumn('fuel_type', regexp_replace('fuel_type', '[^1-7]', '0'))
fue_t9=fue_t8.withColumn('fuel_type', regexp_replace('fuel_type', '(\d{2,3000})', '8'))
fue_t10=fue_t9.withColumn('fuel_type', regexp_replace('fuel_type', '0', '8'))
#fue_t11=fue_t10.withColumn("fuel_type",fue_t10["fuel_type"].cast("double"))
fue_t11=fue_t10.withColumn("fuel_type", col("fuel_type")).na.fill('8')
fue_t12=fue_t11.withColumn('fuel_type', regexp_replace('fuel_type', '1', 'Electric')) 
fue_t13=fue_t12.withColumn('fuel_type', regexp_replace('fuel_type', '2', 'Compressed Natural Gas')) 
fue_t14=fue_t13.withColumn('fuel_type', regexp_replace('fuel_type', '3', 'Hybrid')) 
fue_t15=fue_t14.withColumn('fuel_type', regexp_replace('fuel_type', '4', 'Flex Fuel Vehicle')) 
fue_t16=fue_t15.withColumn('fuel_type', regexp_replace('fuel_type', '5', 'Gasoline')) 
fue_t17=fue_t16.withColumn('fuel_type', regexp_replace('fuel_type', '6', 'Biodiesel')) 
fue_t18=fue_t17.withColumn('fuel_type', regexp_replace('fuel_type', '7', 'Diesel'))
fue_t19=fue_t18.withColumn('fuel_type', regexp_replace('fuel_type', '8', 'Other_fuel'))


# maximum_seating

# In[30]:


max_s1=fue_t19.withColumn('maximum_seating', translate('maximum_seating', 'seats', ''))
max_s2=max_s1.withColumn('maximum_seating', translate('maximum_seating', ' ', ''))
max_s3=max_s2.withColumn('maximum_seating', regexp_replace('maximum_seating', '(\s+\d+\s+)$', '')) 
max_s4=max_s3.withColumn('maximum_seating', regexp_replace('maximum_seating', '[^\d{2}]', '0')) 
max_s5=max_s4.withColumn('maximum_seating', regexp_replace('maximum_seating', '^(\d{2,3000})', '5'))
max_s6=max_s5.withColumn('maximum_seating', regexp_replace('maximum_seating', '0', '5'))
#max_s7=max_s6.withColumn("maximum_seating",max_s6["maximum_seating"].cast("double"))
max_s8=max_s6.withColumn("maximum_seating", col("maximum_seating")).na.fill(5)


# city_fuel_economy

# In[31]:


city1=max_s8.withColumn('city_fuel_economy', regexp_replace('city_fuel_economy', '^(\d{3,3000})', '22'))
city2=city1.withColumn("city_fuel_economy",city1["city_fuel_economy"].cast("double"))
city3=city2.withColumn("city_fuel_economy", col("city_fuel_economy")).na.fill(22)


# owner_count

# In[32]:


oc1=city3.withColumn('owner_count', translate('owner_count', 'in', ''))
oc2=oc1.withColumn('owner_count', translate('owner_count', ' ', ''))
oc3=oc2.withColumn('owner_count', translate('owner_count', '.', ' '))
oc4=oc3.withColumn('owner_count', regexp_replace('owner_Count', '(\s+\d*\s*)$', '')) 
oc5=oc4.withColumn('owner_count', regexp_replace('owner_count', '[^\d{2}]', '0'))
oc6=oc5.withColumn('owner_count', regexp_replace('owner_count', '^(\d{2,2500})', '1'))
oc7=oc6.withColumn('owner_count', translate('owner_count', '0', '2'))
# oc8=oc7.select([when(col('owner_count')=="",None).otherwise(col('owner_count')).alias('owner_count') for c in oc7.columns])
#oc8=oc7.withColumn("owner_count",col("owner_count").cast("double"))
oc9=oc7.withColumn("owner_count", col("owner_count")).na.fill(1)


# wheelbase

# In[33]:


wb1=oc9.withColumn('wheelbase', translate('wheelbase', 'in', ''))
wb2=wb1.withColumn('wheelbase', translate('wheelbase', ' ', ''))
wb3=wb2.withColumn('wheelbase', translate('wheelbase', '.', ' '))
wb4=wb3.withColumn('wheelbase', regexp_replace('wheelbase', '(\s+\d*\s*)$', ''))    #(\s+\d*\s*)$
wb5=wb4.withColumn('wheelbase', regexp_replace('wheelbase', '[^\d{2}]', '0'))
wb6=wb5.withColumn('wheelbase', regexp_replace('wheelbase', '^(\d{4,2500})', '106')) 
wb7=wb6.withColumn('wheelbase', regexp_replace('wheelbase', "(^0+)", "0"))
wb8=wb7.withColumn('wheelbase', regexp_replace('wheelbase', '^0$', '106')) 
#wb8=wb7.select([when(col('wheelbase')=="",None).otherwise(col('wheelbase')).alias('wheel_base') for c in wb7.columns])
wb9 = wb8.withColumn('wheelbase', regexp_replace('wheelbase', r'^[0]*', ''))
wb10 = wb9.withColumn('wheelbase',when(wb9.wheelbase<=9,106).otherwise(wb9.wheelbase))
#wb11 = wb10.withColumn('wheelbase',when(wb10.wheelbase=='',106).otherwise(wb9.wheelbase))
wb11=wb10.withColumn('wheelbase',col('wheelbase').cast("double"))
wb12=wb11.withColumn('wheelbase',col('wheelbase')).na.fill(106)


# horsepower

# In[34]:


hp1=wb12.withColumn('horsepower', translate('horsepower', 'in', ''))
hp2=hp1.withColumn('horsepower', translate('horsepower', ' ', ''))
hp3=hp2.withColumn('horsepower', translate('horsepower', '.', ' '))
hp4=hp3.withColumn('horsepower', regexp_replace('horsepower', '(\s+\d+\s+)$', '')) 
hp5=hp4.withColumn('horsepower', translate('horsepower', ' ', ''))
hp6=hp5.withColumn('horsepower', regexp_replace('horsepower', '[^\d{4}]', '0')) 
hp7=hp6.withColumn('horsepower', regexp_replace('horsepower', '^(\d{4,2500})', '248')) 
hp8=hp7.withColumn('horsepower', regexp_replace('horsepower', '^0*0$', '0'))
hp9=hp8.withColumn('horsepower', translate('horsepower', '0', '248'))
hp10=hp9.withColumn('horsepower', regexp_replace('horsepower', '[^A-Z0-9_]', ''))
hp11=hp10.withColumn('horsepower', regexp_replace('horsepower', '2222[0-9]*', ''))
hp12=hp11.withColumn('horsepower',when(hp11.horsepower<=9,248).otherwise(hp11.horsepower))
hp13=hp12.withColumn('horsepower',col('horsepower').cast("double"))
hp14=hp13.withColumn('horsepower',col('horsepower')).na.fill(248)


# mileage

# In[35]:


mil1=hp14.withColumn('mileage', translate('mileage', 'in', ''))
mil2=mil1.withColumn('mileage', translate('mileage', ' ', ''))
mil3=mil2.withColumn('mileage', regexp_replace('mileage', '(\d+-\d+-\d+)', '')) 
mil4=mil3.withColumn('mileage', translate('mileage', '.', ' '))
mil5=mil4.withColumn('mileage', regexp_replace('mileage', '(\s+\d+\s+)$', '')) 
mil6=mil5.withColumn('mileage', regexp_replace('mileage', '[^\d{6}]', '0')) 
mil7=mil6.withColumn('mileage', regexp_replace('mileage', '^(\d{7,2500})', '31000')) 
mil8=mil7.withColumn('mileage', regexp_replace('mileage', '^0+', '31000')) 
mil9=mil8.withColumn("mileage",col("mileage").cast('double'))
mil10=mil9.withColumn("mileage", col("mileage")).na.fill(31000)


# seller_rating

# In[36]:


sr1=mil10.withColumn('seller_rating', translate('seller_rating', ' ', ''))
sr2=sr1.withColumn('seller_rating', regexp_replace('seller_rating', '[^(\d\d+)]', '0'))
sr3=sr2.withColumn('seller_rating', regexp_replace('seller_rating', '(\s+\d+\s+)$', '')) 
sr4=sr3.withColumn('seller_rating', regexp_replace('seller_rating', '[^\d{1}]', '0'))
sr5=sr4.withColumn('seller_rating', regexp_replace('seller_rating', '^(\d{2,2500})', '4'))
sr6=sr5.withColumn('seller_rating', regexp_replace('seller_rating', '[^1-5$]', '4'))
#sr7=sr6.withColumn('seller_rating',col('seller_rating').cast('double'))
sr8=sr6.withColumn('seller_rating', col('seller_rating')).na.fill(4)


# dealer_zip

# In[37]:


dea1=sr8.withColumn('dealer_zip', regexp_replace('dealer_zip', '(^[A-Za-z]+)', '43228'))
dea2=dea1.withColumn('dealer_zip', regexp_replace('dealer_zip', '[^(\d{3,6})]', '43228'))
dea3=dea2.withColumn("dealer_zip", col("dealer_zip")).na.fill('43228')
#dea4=dea3.withColumn("dealer_zip",col("dealer_zip").cast("double"))
#sr7=sr6.withColumn('seller_rating',col('seller_rating').cast('double'))


# franchise_dealer

# In[38]:


fra1=dea3.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(TRUE)', '1'))
fra2=fra1.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(FALSE)', '0'))
fra3=fra2.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '[^0-1]', '1'))
fra4=fra3.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(\d{2,2500})', '1'))
fra5=fra4.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '1', 'TRUE'))
fra6=fra5.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '0', 'FALSE'))


# In[39]:


fra6.printSchema()


# In[40]:


type(fra6)


# In[41]:


pri1=fra6.filter(col("price").rlike("^[0-9]{3,8}+$"))
#pri2=pri1.withColumn("price",col("price").cast("double"))


# In[42]:


#pri2.toPandas().to_csv('usedcarcleans1.csv')


# In[43]:


#pri1.printSchema()


# # MAPPING COLUMNS

# In[44]:


transmission1=spark.read.parquet("s3://parquetfile07/trans.parquet/")


# In[45]:


# file_location = "s3://projectusedcardatset/transmission_csv.csv"
# file_type = "csv"

# # CSV options
# infer_schema = True
# first_row_is_header = True
# delimiter = ","

# # The applied options are for CSV files. For other file types, these will be ignored.
# transmission1 = spark.read.format(file_type) \
#   .option("inferSchema", infer_schema) \
#   .option("header", first_row_is_header) \
#   .option("sep", delimiter) \
#   .load(file_location)


# In[46]:


transmission1.show()


# In[47]:


bodytype1=spark.read.parquet("s3://parquetfile07/bodyss.parquet/")


# In[48]:


# file_location = "s3://projectusedcardatset/bodytype_csv.csv"
# file_type = "csv"

# # CSV options
# infer_schema = True
# first_row_is_header = True
# delimiter = ","

# # The applied options are for CSV files. For other file types, these will be ignored.
# bodytype1 = spark.read.format(file_type) \
#   .option("inferSchema", infer_schema) \
#   .option("header", first_row_is_header) \
#   .option("sep", delimiter) \
#   .load(file_location)


# In[49]:


bodytype1.show(truncate=False)


# In[50]:


# file_location = "s3://projectusedcardatset/fueltype_csv.csv"
# file_type = "csv"

# # CSV options
# infer_schema = True
# first_row_is_header = True
# delimiter = ","

# # The applied options are for CSV files. For other file types, these will be ignored.
# fueltype1 = spark.read.format(file_type) \
#   .option("inferSchema", infer_schema) \
#   .option("header", first_row_is_header) \
#   .option("sep", delimiter) \
#   .load(file_location)


# In[51]:


fueltype1=spark.read.parquet("s3://parquetfile07/fuelss.parquet/")


# In[52]:


fueltype1.show(truncate=False)


# In[53]:


# file_location = "s3://projectusedcardatset/wheelsystem_csv.csv"
# file_type = "csv"

# # CSV options
# infer_schema = True
# first_row_is_header = True
# delimiter = ","

# # The applied options are for CSV files. For other file types, these will be ignored.
# wheelsystem1 = spark.read.format(file_type) \
#   .option("inferSchema", infer_schema) \
#   .option("header", first_row_is_header) \
#   .option("sep", delimiter) \
#   .load(file_location)


# In[54]:


wheelsystem1=spark.read.parquet("s3://parquetfile07/wheelsys.parquet/")


# In[55]:


wheelsystem1.show()


# In[56]:


# file_location = "s3://projectusedcardatset/enginetype_csv.csv"
# file_type = "csv"

# # CSV options
# infer_schema = True
# first_row_is_header = True
# delimiter = ","

# # The applied options are for CSV files. For other file types, these will be ignored.
# enginetype1 = spark.read.format(file_type) \
#   .option("inferSchema", infer_schema) \
#   .option("header", first_row_is_header) \
#   .option("sep", delimiter) \
#   .load(file_location)


# In[57]:


enginetype1=spark.read.parquet("s3://parquetfile07/enginess.parquet/")


# In[58]:


enginetype1.show()


# In[59]:


fra7=(pri1.join(transmission1,pri1.transmission ==  transmission1.transmission_mode,"inner"))
fra8=(fra7.join(wheelsystem1,fra7.wheel_system == wheelsystem1.wheelsystem_mode,"inner"))
fra9=(fra8.join(fueltype1,fra8.fuel_type ==  fueltype1.fueltype_mode,"inner"))
fra10=(fra9.join(bodytype1,fra9.body_type == bodytype1.bodytype_mode,"inner"))
fra11=(fra10.join(enginetype1,fra10.engine_type ==  enginetype1.enginetype_mode,"inner"))


# In[60]:


# fra11.show(truncate=False)


# In[61]:


fra11.printSchema()


# In[62]:


fra12=fra11.drop('transmission', 'body_type', 'wheel_system', 'engine_type', 'fuel_type', 'transmission_mode', 'bodytype_mode', 
                  'wheelsystem_mode', 'enginetype_mode', 'fueltype_mode')


# In[63]:


fra12.printSchema()


# In[64]:


fra13 = fra12.withColumn('is_new', F.when(fra12.is_new == 'FALSE', 0).otherwise(1)).withColumn('franchise_dealer', F.when(fra12.franchise_dealer == 'FALSE', 0).otherwise(1))


# In[65]:


fra13.printSchema()


# # EXPLORATORY DATA ANALYSIS (EDA)

#  NULL VALUES 

# In[66]:


from pyspark.sql.functions import isnull, when, count, col
data.select([count(when(isnull(c), c)).alias(c) for c in data.columns]).show()


# In[67]:


df=fra13.drop('vin','bed','bed_height','bed_length','cabin','exterior_color','fleet','frame_damaged',
                'has_accidents','interior_color','isCab','power','is_certified','is_cpo','is_oemcpo',
                'listing_id','main_picture_url','make_name','theft_title','vehicle_damage_category',
                'combine_fuel_economy','sp_id','sp_name','trimId','trim_name','wheel_system_display',
                'description','transmission_display','engine_cylinders','salvage','latitude','longitude',
               'savings_amount','franchise_make','listing_color','city', 'model_name', 'listed_date','major_options','year')


# In[68]:


#fran1=list_col1.withColumn('franchise_make', regexp_replace('franchise_make', '(Ford|Chevrolet|Honda|Toyota|Jeep|Hyundai|Nissan|Kia|Subaru|Buick|Volkswagen|Mazda|BMW|GMC|Mercedes-Benz|Volvo|Audi|Cadillac|Dodge|Mitsubishi|Land Rover|Porsche|Jaguar|Maserati| Honda|Ferrari|Porsche| Land Rover|Chevrolet)', '0'))
#fran1.groupby('franchise_make').count().sort('count',ascending=False).collect()


# In[69]:


#lat1=city3.withColumn('latitude', regexp_replace('latitude', '[\d\d\.\d\d\d\d*]', '0'))
#lat1.groupby('latitude').count().sort('count',ascending=False).collect() 


# In[70]:


#list_col1=df.withColumn('listing_color', regexp_replace('listing_color', '(BLACK|WHITE|SILVER|GRAY|BLUE|RED|GREEN|BROWN|YELLOW|GOLD|ORANGE|PURPLE|TEAL|Black (Charcoal)|Gray|jet black|Black (charcoal)|Charcoal Black)', '0'))
#list_col1.groupby('listing_color').count().sort('count',ascending=False).collect() 


# In[71]:


#lon1=lat1.withColumn('longitude', regexp_replace('longitude', '[-\d\d\.\d\d\d\d]', '0'))
#lon1.groupby('longitude').count().sort('count',ascending=False).collect() 


# In[72]:


#sa1 = sr8.withColumn('savings_amount', regexp_replace('savings_amount', '(\d+)', '0'))
#sa1.groupby('savings_amount').count().sort('count', ascending=False).collect()


# OBTAINING SUMMARY STATISTICS

# In[74]:


df.summary().show()


# In[75]:


df.printSchema()


# In[76]:


ra13=df.withColumn("transmission_id",col("transmission_id").cast("double"))
ra14=ra13.withColumn("wheelsystem_id",col("wheelsystem_id").cast("double"))
ra15=ra14.withColumn("fueltype_id",col("fueltype_id").cast("double"))
ra16=ra15.withColumn("bodytype_id",col("bodytype_id").cast("double"))
ra17=ra16.withColumn("enginetype_id",col("enginetype_id").cast("double"))
ra18=ra17.withColumn("is_new",col("is_new").cast("double"))
ra19=ra18.withColumn("franchise_dealer",col("franchise_dealer").cast("double"))
ra20=ra19.withColumn("daysonmarket",col("daysonmarket").cast("double"))
ra21=ra20.withColumn("price",col("price").cast("double"))
ra22=ra21.withColumn("dealer_zip",col("dealer_zip").cast("double"))
ra23=ra22.withColumn('seller_rating',col('seller_rating').cast('double'))
ra24=ra23.withColumn("owner_count",col("owner_count").cast("double"))
ra25=ra24.withColumn('height',col('height').cast("double"))
ra26=ra25.withColumn("maximum_seating",col("maximum_seating").cast("double"))


# In[77]:


ra26.printSchema()


# In[78]:


from pyspark.sql.functions import isnull, when, count, col
ra26.select([count(when(isnull(c), c)).alias(c) for c in ra26.columns]).show()


# VIF

# In[80]:


ra26.columns


# In[89]:


#droping columns due to vif>5

df1=ra26.drop('is_new','height','wheelbase','horsepower','length')


# In[90]:


df1.columns


# CORRELATION
# 

# In[97]:


from pyspark.mllib.stat import Statistics

# select variables to check correlation
df_features = df1.select('back_legroom', 'city_fuel_economy', 'dealer_zip',
                         'engine_displacement', 'front_legroom', 
                         'fuel_tank_volume', 'highway_fuel_economy', 'maximum_seating', 
                         'mileage', 'owner_count', 'price', 'seller_rating', 'torque', 'width')
                          


# In[98]:


#create RDD table for correlation calculation
rdd_table = df_features.rdd.map(lambda row: row[0:])


# In[99]:


#get the correlation matrix
corr_mat=Statistics.corr(rdd_table, method="pearson")
print(corr_mat)


# In[100]:


df2=df1.toPandas()


# # VISUALIZATION

# In[101]:


df2.columns


# In[102]:


df2.hist(column='width', bins=10)


# In[103]:


get_ipython().run_line_magic('matplot', 'plt')


# In[104]:


df2.hist(column='city_fuel_economy', bins=15)


# In[105]:


get_ipython().run_line_magic('matplot', 'plt')


# In[106]:


from pyspark.sql.functions import desc

fig = plt.figure(figsize =(5, 5))


# In[107]:


engDF = df1[['enginetype_id','price']].groupby('enginetype_id').sum('price').sort(desc("sum(price)")).toPandas().head(3)
# display the top 10 organisation group 
plt.bar(engDF["enginetype_id"], engDF["sum(price)"])
plt.title("Top Three Engine type")
plt.xlabel("enginetype_id")
plt.ylabel("price")


# In[108]:


get_ipython().run_line_magic('matplot', 'plt')


# In[109]:


fig = plt.figure(figsize =(5, 5))
scat=df2.plot(kind = 'scatter', x = 'fuel_tank_volume', y = 'price')
plt.show()


# In[110]:


get_ipython().run_line_magic('matplot', 'plt')


# In[111]:



plt.figure();

bp = df2.boxplot(column=['front_legroom'])


# In[112]:


bp


# In[113]:


get_ipython().run_line_magic('matplot', 'plt')


# In[114]:


plt.figure();

bp1 = df2.boxplot(column=['width'])


# In[115]:


bp1


# In[116]:


get_ipython().run_line_magic('matplot', 'plt')


# OUTLIERS

# In[117]:


df1.printSchema()


# In[118]:


va1=df1.withColumn("franchise_dealer",col("franchise_dealer").cast("int"))
va2=va1.withColumn("transmission_id",col("transmission_id").cast("int"))
va3=va2.withColumn("wheelsystem_id",col("wheelsystem_id").cast("int"))
va4=va3.withColumn("fueltype_id",col("fueltype_id").cast("int"))
va5=va4.withColumn("bodytype_id",col("bodytype_id").cast("int"))
va6=va5.withColumn("enginetype_id",col("enginetype_id").cast("int"))
va7=va6.withColumn("price",col("price").cast("int"))
va8=va7.withColumn("maximum_seating",col("maximum_seating").cast("int"))
va9=va8.withColumn("seller_rating",col("seller_rating").cast("int"))
va10=va9.withColumn("owner_count",col("owner_count").cast("int"))


# In[119]:


va10.printSchema()


# In[120]:


def find_outliers(va10):


    numeric_columns = [column[0] for column in va10.dtypes if column[1]=='double']

    
    for column in numeric_columns:

        less_Q1 = 'less_Q1_{}'.format(column)
        more_Q3 = 'more_Q3_{}'.format(column)
        Q1 = 'Q1_{}'.format(column)
        Q3 = 'Q3_{}'.format(column)

        
        Q1 = va10.approxQuantile(column,[0.25],relativeError=0)
        Q3 = va10.approxQuantile(column,[0.75],relativeError=0)
        
        
        IQR = Q3[0] - Q1[0]
        

        less_Q1 =  Q1[0] - 1.5*IQR
        more_Q3 =  Q3[0] + 1.5*IQR
        
        isOutlierCol = 'is_outlier_{}'.format(column)
        
        va10 = va10.withColumn(isOutlierCol,f.when((va10[column] > more_Q3) | (va10[column] < less_Q1), 1).otherwise(0))
    

    
    selected_columns = [column for column in va10.columns if column.startswith("is_outlier")]

    
    va10 = va10.withColumn('total_outliers',sum(va10[column] for column in selected_columns))
    va10 = va10.drop(*[column for column in va10.columns if column.startswith("is_outlier")])

    return va10


# In[121]:


new_df = find_outliers(va10)
new_df.show(5)


# In[122]:


type(new_df)


# In[123]:


new_df.count()


# In[124]:


data_without_outliers = new_df.filter(new_df['total_Outliers']==0)
data_without_outliers.show(5)


# In[125]:


data_without_outliers.count()


# In[ ]:





#  # CHANGING DATA TYPE

# In[126]:


type(data_without_outliers)


# In[127]:


va11=data_without_outliers.withColumn("franchise_dealer",col("franchise_dealer").cast("double"))
va12=va11.withColumn("transmission_id",col("transmission_id").cast("double"))
va13=va12.withColumn("wheelsystem_id",col("wheelsystem_id").cast("double"))
va14=va13.withColumn("fueltype_id",col("fueltype_id").cast("double"))
va15=va14.withColumn("bodytype_id",col("bodytype_id").cast("double"))
va16=va15.withColumn("enginetype_id",col("enginetype_id").cast("double"))
va17=va16.withColumn("price",col("price").cast("double"))
va18=va17.withColumn("maximum_seating",col("maximum_seating").cast("double"))
va19=va18.withColumn("seller_rating",col("seller_rating").cast("double"))
va20=va19.withColumn("owner_count",col("owner_count").cast("double"))


# In[128]:


va20.printSchema()


# In[129]:


z1=va20.drop('total_outliers')


# In[130]:


z1.printSchema()


# In[131]:


z1.columns


# In[132]:


z1.take(1)


# # MODEL

# In[173]:


splits = z1.randomSplit([0.7, 0.3])
trainDF1 = splits[0]
testDF1 = splits[1]


# In[177]:


inpcolumns= ['back_legroom', 'city_fuel_economy', 'daysonmarket', 'dealer_zip', 'engine_displacement',
             'franchise_dealer', 'front_legroom', 'fuel_tank_volume', 'highway_fuel_economy',
             'maximum_seating', 'mileage', 'owner_count', 'seller_rating', 'torque',
             'width', 'transmission_id', 'wheelsystem_id', 'fueltype_id', 'bodytype_id', 'enginetype_id']


# In[178]:


assembler = VectorAssembler(inputCols= inpcolumns, outputCol= "features")


# In[179]:


desicion_tree_reg = DecisionTreeRegressor(featuresCol="features", labelCol="price", maxDepth=10)


# In[180]:


pipe=Pipeline(stages=[assembler,desicion_tree_reg])


# In[181]:


final_pipeline=pipe.fit(trainDF1)


# In[182]:


final_pipeline.save("s3://carparq/pick")


# In[183]:


persistedModel = final_pipeline.load("s3://carparq/pick")


# In[184]:


predictionsDT11 = persistedModel.transform(testDF1)


# In[185]:


predictionsDT11.select( "price", "prediction").take(5)


# # EVALUATION

# In[186]:


from pyspark.ml.evaluation import RegressionEvaluator
dt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(predictionsDT11)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[187]:


dt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="r2")
r2 = dt_evaluator.evaluate(predictionsDT11)
print("R2  on test data = %g" % r2)


# In[188]:


dt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="mae")
mae = dt_evaluator.evaluate(predictionsDT11)
print("Mean Absolute Error (MAE) on test data = %g" % mae)


# In[189]:


dt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="mse")
mse = dt_evaluator.evaluate(predictionsDT11)
print("Mean square Error (MsE) on test data = %g" % mse)


# # PREDICTION

# In[190]:


columns=['back_legroom', 'city_fuel_economy', 'daysonmarket', 'dealer_zip', 'engine_displacement',
             'franchise_dealer', 'front_legroom', 'fuel_tank_volume', 'highway_fuel_economy',
             'maximum_seating', 'mileage', 'owner_count', 'seller_rating', 'torque',
             'width', 'transmission_id', 'wheelsystem_id', 'fueltype_id', 'bodytype_id', 'enginetype_id']


# In[191]:


input=[(38.0,22.0,6.0,10977.0,2970.0,1.0,40.0,26.0,30.0,5.0,5.0,1.0,4.0,250.0,73.0,2.0,3.0,8.0,5.0,7.0)]


# In[192]:


#input1=[(35.0,22.0,522.0,960.0,1300.0,1.0,41.0,12.0,30.0,5.0,7.0,1.0,4.0,200.0,79.0,'A','FWD','Gasoline','SUV / Crossover','I4')]


# In[193]:


input_dataframe= spark.createDataFrame(input, columns)


# In[194]:


file_location = "s3://projectusedcardatset/transmission_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
transmission1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[195]:


file_location = "s3://projectusedcardatset/bodytype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
bodytype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[196]:


file_location = "s3://projectusedcardatset/fueltype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
fueltype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[197]:


file_location = "s3://projectusedcardatset/wheelsystem_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
wheelsystem1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[198]:


file_location = "s3://projectusedcardatset/enginetype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
enginetype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[199]:


# a7=input_dataframe.join(transmission1,input_dataframe.transmission_id ==  transmission1.transmission_mode,"inner")
# a8=a7.join(wheelsystem1,a7.wheelsystem_id == wheelsystem1.wheelsystem_mode,"inner")
# a9=a8.join(fueltype1,a8.fueltype_id  ==  fueltype1.fueltype_mode,"inner")
# a10=a9.join(bodytype1,a9.bodytype_id  == bodytype1.bodytype_mode,"inner")
# a11=a10.join(enginetype1,a10.enginetype_id ==  enginetype1.enginetype_mode,"inner")


# In[200]:


#a12 = a11.withColumn('is_new', F.when(fra17.is_new == 'FALSE', 0).otherwise(1)).withColumn('franchise_dealer', F.when(fra17.franchise_dealer == 'FALSE', 0).otherwise(1))


# In[201]:


#a12.printSchema()


# In[202]:


#a13=a12.drop('transmission','body_type', 'wheel_system', 'engine_type', 'fuel_type', 'transmission_mode', 'bodytype_mode', 'wheelsystem_mode', 'enginetype_mode', 'fueltype_mode')


# In[203]:


#a13.printSchema()


# In[204]:


# a14=a13.withColumn("transmission_id",col("transmission_id").cast("double"))
# a15=a14.withColumn("wheelsystem_id",col("wheelsystem_id").cast("double"))
# a16=a15.withColumn("fueltype_id ",col("fueltype_id ").cast("double"))
# a17=a16.withColumn("bodytype_id ",col("bodytype_id ").cast("double"))
# a18=a17.withColumn("enginetype_id",col("enginetype_id").cast("double"))
# a19=a18.withColumn("is_new",col("is_new").cast("double"))
# a20=a19.withColumn("franchise_dealer",col("franchise_dealer").cast("double"))


# In[205]:


#a20.printSchema()


# In[206]:


predictionsDT12 = persistedModel.transform(input_dataframe)


# In[207]:


predictionsDT12.select( "prediction").show()


# In[208]:


#input_dataframe_result = final_pipeline.transform(input_dataframe)


# In[209]:


#input_dataframe_result.select( "prediction").show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




