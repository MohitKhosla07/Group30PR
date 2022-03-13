#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark.sql import SparkSession
import warnings
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pyspark.sql.functions import when, lit, col, substring, substring_index
import pyspark.sql.functions
from pyspark.sql.functions import translate
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler,OneHotEncoder, StringIndexer


# In[ ]:


spark = SparkSession.builder.appName("Spark").getOrCreate()


# In[ ]:


file_location = "C:\\Users\\shivs\\Desktop\\ZFINALPROJECT\\used_cars_data (1).csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
data = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[ ]:


file_location = "transmission_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
transmission1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)
#transmission1.show()


# In[ ]:


file_location = "bodytype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
bodytype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)
#bodytype1.show(truncate=False)


# In[ ]:


file_location = "fueltype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
fueltype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)
#fueltype1.show(truncate=False)


# In[ ]:


file_location = "wheelsystem_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
wheelsystem1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)

#wheelsystem1.show()


# In[ ]:


file_location = "enginetype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
enginetype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)

#enginetype1.show()


# In[ ]:


type(data)


# In[ ]:


data.count()


# In[ ]:


len(data.columns)


# In[ ]:


data.printSchema()


# In[ ]:


from pyspark.sql.functions import isnull, when, count, col
data.select([count(when(isnull(c), c)).alias(c) for c in data.columns]).show()


# In[ ]:


data.columns


# In[ ]:


df=data.drop('vin','bed','bed_height','bed_length','cabin','exterior_color','fleet','frame_damaged',
                 'has_accidents','interior_color','isCab','power',
                 'is_certified','is_cpo','is_oemcpo',
                'listing_id','main_picture_url','make_name',
                'theft_title','vehicle_damage_category','combine_fuel_economy','sp_id',
                'sp_name','trimId','trim_name','wheel_system_display','description',
              'transmission_display','engine_cylinders','salvage','latitude','longitude','savings_amount','franchise_make',
         'listing_color','city', 'model_name', 'listed_date','major_options')


# In[ ]:


len(df.columns)


# In[ ]:


df.printSchema()


# In[ ]:


#WIDTH

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


#IS_NEW
is_n1=wid10.withColumn('is_new', regexp_replace('is_new', '(TRUE)', '1'))
is_n2=is_n1.withColumn('is_new', regexp_replace('is_new', '(FALSE)', '0'))
is_n3=is_n2.withColumn('is_new', regexp_replace('is_new', '[^0-1]', '0'))
is_n4=is_n3.withColumn('is_new', regexp_replace('is_new', '(\d{2,2500})', '0'))
is_n5=is_n4.withColumn('is_new', regexp_replace('is_new', '1', 'TRUE'))
is_n6=is_n5.withColumn('is_new', regexp_replace('is_new', '0', 'FALSE'))
is_n7=is_n6.withColumn('is_new', col('is_new')).na.fill('TRUE')
# is_n7.groupby('is_new').count().sort('count', ascending=False).collect()


# In[ ]:


sa1 = sr8.withColumn('savings_amount', regexp_replace('savings_amount', '(\d+)', '0'))
sa1.groupby('savings_amount').count().sort('count', ascending=False).collect()


# In[ ]:


dea1=sr8.withColumn('dealer_zip', regexp_replace('dealer_zip', '(^[A-Za-z]+)', '43228'))
dea2=dea1.withColumn('dealer_zip', regexp_replace('dealer_zip', '[^(\d{3,6})]', '43228'))
dea3=dea2.withColumn("dealer_zip", col("dealer_zip")).na.fill('43228')
dea4=dea3.withColumn("dealer_zip",col("dealer_zip").cast("double"))


# In[ ]:


fra1=dea4.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(TRUE)', '1'))
fra2=fra1.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(FALSE)', '0'))
fra3=fra2.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '[^0-1]', '1'))
fra4=fra3.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(\d{2,2500})', '1'))
fra5=fra4.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '1', 'TRUE'))
fra6=fra5.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '0', 'FALSE'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




