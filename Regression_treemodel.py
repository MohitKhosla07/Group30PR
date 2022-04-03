#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install pyspark


# In[6]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[1]:


from pyspark.sql import SparkSession
# import warnings
# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns
from pyspark.sql.functions import when, lit, col, substring, substring_index
import pyspark.sql.functions
from pyspark.sql.functions import translate
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler,OneHotEncoder, StringIndexer


# In[2]:


spark = SparkSession.builder.appName("Spark").getOrCreate()


# In[3]:


file_location = "s3://gauravdata/used_cars_data.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
data = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[4]:


# type(data)


# In[5]:


# data.count()


# In[6]:


# len(data.columns)


# In[7]:


# data.printSchema()


# In[8]:


#from pyspark.sql.functions import isnull, when, count, col
#data.select([count(when(isnull(c), c)).alias(c) for c in data.columns]).show()


# In[9]:


data.columns


# In[10]:


df=data.drop('vin','bed','bed_height','bed_length','cabin','exterior_color','fleet','frame_damaged',
                 'has_accidents','interior_color','isCab','power',
                 'is_certified','is_cpo','is_oemcpo',
                'listing_id','main_picture_url','make_name',
                'theft_title','vehicle_damage_category','combine_fuel_economy','sp_id',
                'sp_name','trimId','trim_name','wheel_system_display','description',
              'transmission_display','engine_cylinders','salvage','latitude','longitude','savings_amount','franchise_make',
         'listing_color','city', 'model_name', 'listed_date','major_options')


# In[11]:


len(df.columns)


# In[12]:


df.printSchema()


# In[13]:


# len(df.columns)


# In[ ]:





# width

# In[14]:


wid1=df.withColumn('width', translate('width', 'in', ''))
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


# In[ ]:





# Year

# In[ ]:





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


# In[ ]:





# Torque

# In[16]:


tor1=is_n7.withColumn('torque', translate('torque', ' ', ''))
tor2 = tor1.withColumn('torque', substring('torque', 1,3))
tor3=tor2.withColumn('torque', regexp_replace('torque', '[^\d{3}]', '0'))
tor4=tor3.withColumn('torque', regexp_replace('torque', '^0[0-4][0-9]', '250'))
tor5=tor4.withColumn('torque', regexp_replace('torque', '^(\d{1,2})$', '250'))
tor6=tor5.withColumn("torque",col("torque").cast("double"))
tor7=tor6.withColumn("torque", col("torque")).na.fill(250)


# Daysonmarket

# In[17]:


day1=tor7.withColumn("daysonmarket",col("daysonmarket").cast("double"))
day2=day1.withColumn("daysonmarket", col("daysonmarket")).na.fill(76)


# Engine_Displacement

# In[18]:


engd1=day2.withColumn('engine_displacement', regexp_replace('engine_displacement', '[^\d{4}]', '0')) 
engd2=engd1.withColumn('engine_displacement', regexp_replace('engine_displacement', '^(\d{5,3500})', '2970'))
engd3=engd2.withColumn('engine_displacement', regexp_replace('engine_displacement', '^0[0-9]*$', '2970'))
engd4=engd3.withColumn('engine_displacement', regexp_replace('engine_displacement', '^[0-9]$', '2970'))
engd5=engd4.withColumn("engine_displacement",col("engine_displacement").cast("double"))
engd6=engd5.withColumn("engine_displacement", col("engine_displacement")).na.fill(2970)


# Highway_fuel_economy

# In[19]:


hig1=engd6.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '[^\d{3}]', '0')) 
hig2 = hig1.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '(\d{4,3000})', '0')) 
hig3 = hig2.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '^0', '30'))
hig4 = hig3.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '30[0-9]*$', '30'))
hig5=hig4.withColumn("highway_fuel_economy",col("highway_fuel_economy").cast("double"))
hig6=hig5.withColumn("highway_fuel_economy", col("highway_fuel_economy")).na.fill(30)


# Wheel_system

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


# 'Transmission'

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


# 'back_legroom'

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


# 'front_legroom'

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


# 'height'

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
hei11=hei10.withColumn('height',col('height').cast("double"))
hei12=hei11.withColumn('height',col('height')).na.fill(66)


# 'length'

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


# 'fuel_tank_volume'

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


# 'body_type'

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


# 'engine_type'

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


# 'fuel_type'

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
max_s7=max_s6.withColumn("maximum_seating",max_s6["maximum_seating"].cast("double"))
max_s8=max_s7.withColumn("maximum_seating", col("maximum_seating")).na.fill(5)


# 'city_fuel_economy'

# In[31]:


city1=max_s8.withColumn('city_fuel_economy', regexp_replace('city_fuel_economy', '^(\d{3,3000})', '22'))
city2=city1.withColumn("city_fuel_economy",city1["city_fuel_economy"].cast("double"))
city3=city2.withColumn("city_fuel_economy", col("city_fuel_economy")).na.fill(22)


# 'Owner_count'

# In[32]:


oc1=city3.withColumn('owner_count', translate('owner_count', 'in', ''))
oc2=oc1.withColumn('owner_count', translate('owner_count', ' ', ''))
oc3=oc2.withColumn('owner_count', translate('owner_count', '.', ' '))
oc4=oc3.withColumn('owner_count', regexp_replace('owner_Count', '(\s+\d*\s*)$', '')) 
oc5=oc4.withColumn('owner_count', regexp_replace('owner_count', '[^\d{2}]', '0'))
oc6=oc5.withColumn('owner_count', regexp_replace('owner_count', '^(\d{2,2500})', '1'))
oc7=oc6.withColumn('owner_count', translate('owner_count', '0', '2'))
# oc8=oc7.select([when(col('owner_count')=="",None).otherwise(col('owner_count')).alias('owner_count') for c in oc7.columns])
oc8=oc7.withColumn("owner_count",col("owner_count").cast("double"))
oc9=oc8.withColumn("owner_count", col("owner_count")).na.fill(1)


# 'Wheelbase'

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


# 'Horse_power'

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


# 'Mileage'

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


# 'Seller_Rating'

# In[36]:


sr1=mil10.withColumn('seller_rating', translate('seller_rating', ' ', ''))
sr2=sr1.withColumn('seller_rating', regexp_replace('seller_rating', '[^(\d\d+)]', '0'))
sr3=sr2.withColumn('seller_rating', regexp_replace('seller_rating', '(\s+\d+\s+)$', '')) 
sr4=sr3.withColumn('seller_rating', regexp_replace('seller_rating', '[^\d{1}]', '0'))
sr5=sr4.withColumn('seller_rating', regexp_replace('seller_rating', '^(\d{2,2500})', '4'))
sr6=sr5.withColumn('seller_rating', regexp_replace('seller_rating', '[^1-5$]', '4'))
sr7=sr6.withColumn('seller_rating',col('seller_rating').cast('double'))
sr8=sr7.withColumn('seller_rating', col('seller_rating')).na.fill(4)


# 'dealer_zip'

# In[37]:


dea1=sr8.withColumn('dealer_zip', regexp_replace('dealer_zip', '(^[A-Za-z]+)', '43228'))
dea2=dea1.withColumn('dealer_zip', regexp_replace('dealer_zip', '[^(\d{3,6})]', '43228'))
dea3=dea2.withColumn("dealer_zip", col("dealer_zip")).na.fill('43228')
dea4=dea3.withColumn("dealer_zip",col("dealer_zip").cast("double"))


# 'Franchise_dealer'

# In[38]:


fra1=dea4.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(TRUE)', '1'))
fra2=fra1.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(FALSE)', '0'))
fra3=fra2.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '[^0-1]', '1'))
fra4=fra3.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(\d{2,2500})', '1'))
fra5=fra4.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '1', 'TRUE'))
fra6=fra5.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '0', 'FALSE'))


# Type

# In[39]:


fra6.printSchema()


# In[40]:


type(fra6)


# price

# In[41]:


pri1=fra6.filter(col("price").rlike("^[0-9]{3,8}+$"))
pri2=pri1.withColumn("price",col("price").cast("double"))


# In[ ]:





# In[ ]:





# convert_categorical_to_dataframe

# In[42]:


file_location = "s3://gauravdata/transmission_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
transmission1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[43]:


transmission1.show()


# In[44]:


file_location = "s3://gauravdata/Bodytype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
bodytype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[45]:


bodytype1.show(truncate=False)


# In[46]:


file_location = "s3://gauravdata/fueltype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
fueltype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[47]:


fueltype1.show(truncate=False)


# In[48]:


file_location = "s3://gauravdata/wheelsystem_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
wheelsystem1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[49]:


wheelsystem1.show()


# In[50]:


wheelsystem1.printSchema()


# In[51]:


file_location = "s3://gauravdata/enginetype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
enginetype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[52]:


enginetype1.show()


# In[53]:


fra7=(pri2.join(transmission1,fra6.transmission ==  transmission1.transmission_mode,"inner"))
fra8=(fra7.join(wheelsystem1,fra7.wheel_system == wheelsystem1.wheelsystem_mode,"inner"))
fra9=(fra8.join(fueltype1,fra8.fuel_type ==  fueltype1.fueltype_mode,"inner"))
fra10=(fra9.join(bodytype1,fra9.body_type == bodytype1.bodytype_mode,"inner"))
fra11=(fra10.join(enginetype1,fra10.engine_type ==  enginetype1.enginetype_mode,"inner"))


# In[54]:


# fra11.show(truncate=False)


# In[55]:


fra11.printSchema()


# In[56]:


fra12=fra11.drop('transmission', 'body_type', 'wheel_system', 'engine_type', 'fuel_type', 'transmission_mode', 'bodytype_mode', 
                  'wheelsystem_mode', 'enginetype_mode', 'fueltype_mode')


# In[57]:


fra12.printSchema()


# In[58]:


fra13=fra12.withColumn("transmission_id",col("transmission_id").cast("double"))


# In[59]:


fra14=fra13.withColumn("wheelsystem_id",col("wheelsystem_id").cast("double"))


# In[60]:


fra15=fra14.withColumn("fueltype_id ",col("fueltype_id ").cast("double"))


# In[61]:


fra16=fra15.withColumn("bodytype_id ",col("bodytype_id ").cast("double"))


# In[62]:


fra17=fra16.withColumn("enginetype_id",col("enginetype_id").cast("double"))


# In[63]:


fra17.printSchema()


# Feature Engineering

# In[64]:


is_new_indexer = StringIndexer(inputCol="is_new", outputCol="is_newIndex")
fra18 = is_new_indexer.fit(fra17).transform(fra17)


# In[65]:


franchise_dealer_indexer = StringIndexer(inputCol="franchise_dealer", outputCol="franchise_dealerIndex")
fra19 = franchise_dealer_indexer.fit(fra18).transform(fra18)


# In[66]:


# fra19.groupby('is_newIndex').count().sort('count', ascending=False).collect()


# In[67]:


# fra19.groupby('franchise_dealerIndex').count().sort('count', ascending=False).collect()


# In[68]:


fra20=fra19.drop('franchise_dealer', 'is_new', 'year')


# In[69]:


fra20.printSchema()


# In[70]:


inpcolumns= ['back_legroom', 'city_fuel_economy', 'daysonmarket', 'dealer_zip',
       'engine_displacement', 'front_legroom', 'fuel_tank_volume', 'height',
       'highway_fuel_economy', 'horsepower', 'length', 'maximum_seating',
       'mileage', 'owner_count', 'price', 'seller_rating', 'torque',
       'wheelbase', 'width',  'transmission_id', 'wheelsystem_id',
       'fueltype_id ', 'bodytype_id ', 'enginetype_id', 'is_newIndex',
       'franchise_dealerIndex']


# In[71]:


# Import VectorAssembler from pyspark.ml.feature package
from pyspark.ml.feature import VectorAssembler
# Create a list of all the variables that you want to create feature vectors
# These features are then further used for training model

# Create the VectorAssembler object
assembler = VectorAssembler(inputCols= inpcolumns, outputCol= "features")
assembledDF = assembler.transform(fra20)
assembledDF.select("features").show(5, False)


# In[72]:


assembledDF1=assembledDF.select("features", "price")


# In[73]:


assembledDF1.columns


# Convert sparse vector to dense vector

# In[74]:


from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT

# Define a udf that converts sparse vector into dense vector
# You cannot create your own custom function and run that against the data directly. 
# In Spark, You have to register the function first using udf function
sparseToDense = F.udf(lambda v : Vectors.dense(v), VectorUDT())

# We then call the function here passing the column name on which the function has to be applied
densefeatureDF = assembledDF1.withColumn('features_array', sparseToDense('features'))

# densefeatureDF.select("features", "features_array").show(5, False)


# In[75]:


densefeatureDF.select("features").show(5, False)


# In[76]:


densefeatureDF.select("price").show(5, False)


# In[77]:


densefeatureDF.columns


# In[78]:


densefeatureDF1=densefeatureDF.drop("features")


# In[79]:


# trainDF, testDF =  stdscaledDF1.randomSplit([0.7,0.3], seed = 2020)
splits = assembledDF1.randomSplit([0.7, 0.3])
trainDF1 = splits[0]
testDF1 = splits[1]

# print the count of observations in each set
# print("Observations in training set = ", trainDF1.count())
# print("Observations in testing set = ", testDF1.count())


# desicion tree

# In[80]:


from pyspark.ml.regression import DecisionTreeRegressor


# In[81]:


from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator


# In[82]:


from pyspark.ml.evaluation import RegressionEvaluator


# In[83]:


desicion_tree_reg = DecisionTreeRegressor(featuresCol="features", labelCol="price")


# In[84]:


pred_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="mae")


# In[85]:


paramGrid = ParamGridBuilder()     .addGrid(desicion_tree_reg.maxDepth, [1, 5, 10, 15, 20])     .build()


crossval = CrossValidator(estimator=desicion_tree_reg,
                          estimatorParamMaps=paramGrid,
                          evaluator=pred_evaluator,
                          numFolds=5)


# In[86]:


cvModel1 = crossval.fit(assembledDF1)


# In[87]:


cvModel1.bestModel


# In[88]:


cvModel1.bestModel


# In[89]:


cvModel1.avgMetrics   # for desicion tree depth 1 5 10


# In[89]:


cvModel1.avgMetrics


# In[110]:


predictionDT = cvModel1.transform(testDF1)


# In[85]:


# predictionsDT = desicion_tree.transform(testDF1)


# In[ ]:


# predictionsDT.columns


# In[ ]:


# predictionsDT.select("price", "prediction").take(5)


# In[90]:


# predictionsDT.select("price", "prediction").take(20)


# In[112]:


from pyspark.ml.evaluation import RegressionEvaluator
dt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(predictionDT)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[113]:


dt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="r2")
r2 = dt_evaluator.evaluate(predictionDT)
print("R2  on test data = %g" % r2)


# In[115]:


dt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="mae")
mae = dt_evaluator.evaluate(predictionDT)
print("Mean Absolute Error (MAE) on test data = %g" % mae)


# In[116]:


dt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="mse")
mse = dt_evaluator.evaluate(predictionDT)
print("Mean square Error (MsE) on test data = %g" % mse)


# In[ ]:





# In[ ]:





# Random_forest

# In[ ]:





# In[ ]:





# In[ ]:





# In[90]:


from  pyspark.ml.regression import RandomForestRegressor


# In[94]:


random_forest_reg = RandomForestRegressor(featuresCol="features", labelCol="price", numTrees=25)


# In[95]:


model =  random_forest_reg.fit(trainDF1)


# In[96]:


predictions1 = model.transform(testDF1)


# In[97]:


predictions1.columns


# In[98]:


from pyspark.ml.evaluation import RegressionEvaluator
pred_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="r2")
print("R squared (R2) on test data =", pred_evaluator.evaluate(predictions1))


# In[99]:


pred_evaluator1 = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
print("Root Mean Squared Error (RMSE) on test data =", pred_evaluator1.evaluate(predictions1))


# In[100]:


pred_evaluator1 = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="mae")
print(" Mean absolute Error (MaE) on test data =", pred_evaluator1.evaluate(predictions1))


# In[ ]:


predictions1.select("prediction", "price").take(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[101]:


rf = RandomForestRegressor(featuresCol="features", labelCol="price")


# In[102]:


paramGrid = ParamGridBuilder()     .addGrid(rf.maxDepth, [1, 5, 10, 15, 20])     .build()


crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=pred_evaluator,
                          numFolds=2)


# In[ ]:


cvModel1 = crossval.fit(assembledDF1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Gradient_Bossting

# In[ ]:


from pyspark.ml.regression import GBTRegressor


# In[ ]:


gbt = GBTRegressor(featuresCol = 'scaledfeatures', labelCol = 'price', maxIter=10)
gbt_model = gbt.fit(trainDF)


# In[ ]:


gbt_predictions = gbt_model.transform(testDF)
# gbt_predictions.select('prediction', 'price', 'scaledfeatures').show(5)


# In[ ]:


gbt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = gbt_evaluator.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[ ]:


gbt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="r2")
r2 = gbt_evaluator.evaluate(gbt_predictions)
print("R2  on test data = %g" % r2)


# In[ ]:


gbt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="mae")
mae = gbt_evaluator.evaluate(gbt_predictions)
print("Mean Absolute Error (MAE) on test data = %g" % mae)


# In[ ]:


gbt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="mse")
mse = gbt_evaluator.evaluate(gbt_predictions)
print("Mean Squared Error (MSE) on test data = %g" % mse)


# In[ ]:


gbt_predictions.select('prediction', 'price').take(5)


# In[ ]:


gbt_predictions.take(1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# franchise_dealer_vector = OneHotEncoder(inputCol="franchise_dealerIndex", outputCol="franchise_dealer_vec")


# In[ ]:


# from pyspark.ml.feature import OneHotEncoder
# encoder = OneHotEncoder(inputCols= ["gender_indexed", 'heart_disease_indexed', 'smoking_history_indexed'], 
#                          outputCols=["genderVec", 'heart_diseaseVec', 'smoking_historyVec'])
# encodedDF = encoder.fit(strindexedDF).transform(strindexedDF)
# encodedDF.select('gender_indexed', 'genderVec', 'heart_disease_indexed', 'heart_diseaseVec', 
#                     'smoking_history_indexed', 'smoking_historyVec',).show(5, False)


# # EDA

# In[ ]:


import pyspark.pandas as ps


# In[ ]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Spark").getOrCreate()


# In[ ]:


from pyspark.sql import SparkSession
builder = SparkSession.builder.appName("pandas-on-spark")
#builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
# Pandas API on Spark automatically uses this Spark session with the configurations set.
builder.getOrCreate()


# In[ ]:


pd=fra20.to_pandas_on_spark()


# In[ ]:


pd.columns


# In[ ]:


columns= ['back_legroom', 'city_fuel_economy', 'daysonmarket', 'dealer_zip',
       'engine_displacement', 'front_legroom', 'fuel_tank_volume', 'height',
       'highway_fuel_economy', 'horsepower', 'length', 'maximum_seating',
       'mileage', 'owner_count', 'price', 'seller_rating', 'torque',
       'wheelbase', 'width',  'transmission_id', 'wheelsystem_id',
       'fueltype_id ', 'bodytype_id ', 'enginetype_id', 'is_newIndex',
       'franchise_dealerIndex']


# In[ ]:


pd.dtypes


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 'listing_color'

# In[ ]:


#list_col1=city3.withColumn('listing_color', regexp_replace('listing_color', '(BLACK|WHITE|SILVER|GRAY|BLUE|RED|GREEN|BROWN|YELLOW|GOLD|ORANGE|PURPLE|TEAL|Black (Charcoal)|Gray|jet black|Black (charcoal)|Charcoal Black)', '0'))
#list_col1.groupby('listing_color').count().sort('count',ascending=False).collect() 


# 'franchise_make'

# In[ ]:


#fran1=list_col1.withColumn('franchise_make', regexp_replace('franchise_make', '(Ford|Chevrolet|Honda|Toyota|Jeep|Hyundai|Nissan|Kia|Subaru|Buick|Volkswagen|Mazda|BMW|GMC|Mercedes-Benz|Volvo|Audi|Cadillac|Dodge|Mitsubishi|Land Rover|Porsche|Jaguar|Maserati| Honda|Ferrari|Porsche| Land Rover|Chevrolet)', '0'))
#fran1.groupby('franchise_make').count().sort('count',ascending=False).collect()


# 'latitude'

# In[ ]:


#lat1=city3.withColumn('latitude', regexp_replace('latitude', '[\d\d\.\d\d\d\d*]', '0'))
#lat1.groupby('latitude').count().sort('count',ascending=False).collect() 


# 'longitude'

# In[ ]:


#lon1=lat1.withColumn('longitude', regexp_replace('longitude', '[-\d\d\.\d\d\d\d]', '0'))
#lon1.groupby('longitude').count().sort('count',ascending=False).collect() 


# 'Savings_Amount'

# In[ ]:


#sa1 = sr8.withColumn('savings_amount', regexp_replace('savings_amount', '(\d+)', '0'))
#sa1.groupby('savings_amount').count().sort('count', ascending=False).collect()

