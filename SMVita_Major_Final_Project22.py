#!/usr/bin/env python
# coding: utf-8




get_ipython().system('pip install pyspark')





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





spark = SparkSession.builder.appName("Spark").getOrCreate()





file_location = "C:\\Users\\shivs\\Desktop\\ZFINALPROJECT\\used_cars_data (1).csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
data = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)





file_location = "transmission_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
transmission1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)
#transmission1.show()





file_location = "bodytype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
bodytype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)
#bodytype1.show(truncate=False)





file_location = "fueltype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
fueltype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)
#fueltype1.show(truncate=False)





file_location = "wheelsystem_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
wheelsystem1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)

#wheelsystem1.show()





file_location = "enginetype_csv.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = True
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
enginetype1 = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)

#enginetype1.show()





type(data)





data.count()





len(data.columns)





data.printSchema()





from pyspark.sql.functions import isnull, when, count, col
data.select([count(when(isnull(c), c)).alias(c) for c in data.columns]).show()





data.columns





df=data.drop('vin','bed','bed_height','bed_length','cabin','exterior_color','fleet','frame_damaged',
                 'has_accidents','interior_color','isCab','power',
                 'is_certified','is_cpo','is_oemcpo',
                'listing_id','main_picture_url','make_name',
                'theft_title','vehicle_damage_category','combine_fuel_economy','sp_id',
                'sp_name','trimId','trim_name','wheel_system_display','description',
              'transmission_display','engine_cylinders','salvage','latitude','longitude','savings_amount','franchise_make',
         'listing_color','city', 'model_name', 'listed_date','major_options')




len(df.columns)





df.printSchema()





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





#IS_NEW

is_n1=wid10.withColumn('is_new', regexp_replace('is_new', '(TRUE)', '1'))
is_n2=is_n1.withColumn('is_new', regexp_replace('is_new', '(FALSE)', '0'))
is_n3=is_n2.withColumn('is_new', regexp_replace('is_new', '[^0-1]', '0'))
is_n4=is_n3.withColumn('is_new', regexp_replace('is_new', '(\d{2,2500})', '0'))
is_n5=is_n4.withColumn('is_new', regexp_replace('is_new', '1', 'TRUE'))
is_n6=is_n5.withColumn('is_new', regexp_replace('is_new', '0', 'FALSE'))
is_n7=is_n6.withColumn('is_new', col('is_new')).na.fill('TRUE')
# is_n7.groupby('is_new').count().sort('count', ascending=False).collect()


#savings_amount

#sa1 = is_n7.withColumn('savings_amount', regexp_replace('savings_amount', '(\d+)', '0'))
#sa1.groupby('savings_amount').count().sort('count', ascending=False).collect()


#dealer_zip

dea1=is_n7.withColumn('dealer_zip', regexp_replace('dealer_zip', '(^[A-Za-z]+)', '43228'))
dea2=dea1.withColumn('dealer_zip', regexp_replace('dealer_zip', '[^(\d{3,6})]', '43228'))
dea3=dea2.withColumn("dealer_zip", col("dealer_zip")).na.fill('43228')
dea4=dea3.withColumn("dealer_zip",col("dealer_zip").cast("double"))


#franchise_dealer


fra1=dea4.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(TRUE)', '1'))
fra2=fra1.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(FALSE)', '0'))
fra3=fra2.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '[^0-1]', '1'))
fra4=fra3.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '(\d{2,2500})', '1'))
fra5=fra4.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '1', 'TRUE'))
fra6=fra5.withColumn('franchise_dealer', regexp_replace('franchise_dealer', '0', 'FALSE'))


#owner_count

oc1=fra6.withColumn('owner_count', translate('owner_count', 'in', ''))
oc2=oc1.withColumn('owner_count', translate('owner_count', ' ', ''))
oc3=oc2.withColumn('owner_count', translate('owner_count', '.', ' '))
oc4=oc3.withColumn('owner_count', regexp_replace('owner_Count', '(\s+\d*\s*)$', '')) 
oc5=oc4.withColumn('owner_count', regexp_replace('owner_count', '[^\d{2}]', '0'))
oc6=oc5.withColumn('owner_count', regexp_replace('owner_count', '^(\d{2,2500})', '1'))
oc7=oc6.withColumn('owner_count', translate('owner_count', '0', '2'))
# oc8=oc7.select([when(col('owner_count')=="",None).otherwise(col('owner_count')).alias('owner_count') for c in oc7.columns])
oc8=oc7.withColumn("owner_count",col("owner_count").cast("double"))
oc9=oc8.withColumn("owner_count", col("owner_count")).na.fill(1)



#wheel_base

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



#horsepower

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



#Mileage

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



#seller_rating

sr1=mil10.withColumn('seller_rating', translate('seller_rating', ' ', ''))
sr2=sr1.withColumn('seller_rating', regexp_replace('seller_rating', '[^(\d\d+)]', '0'))
sr3=sr2.withColumn('seller_rating', regexp_replace('seller_rating', '(\s+\d+\s+)$', '')) 
sr4=sr3.withColumn('seller_rating', regexp_replace('seller_rating', '[^\d{1}]', '0'))
sr5=sr4.withColumn('seller_rating', regexp_replace('seller_rating', '^(\d{2,2500})', '4'))
sr6=sr5.withColumn('seller_rating', regexp_replace('seller_rating', '[^1-5$]', '4'))
sr7=sr6.withColumn('seller_rating',col('seller_rating').cast('double'))
sr8=sr7.withColumn('seller_rating', col('seller_rating')).na.fill(4)


pri1=sr8.filter(col("price").rlike("^[0-9]{3,8}+$"))
pri2=pri1.withColumn("price",col("price").cast("double"))


#fran1=sr8.withColumn('franchise_make', regexp_replace('franchise_make', '(Ford|Chevrolet|Honda|Toyota|Jeep|Hyundai|Nissan|Kia|Subaru|Buick|Volkswagen|Mazda|BMW|GMC|Mercedes-Benz|Volvo|Audi|Cadillac|Dodge|Mitsubishi|Land Rover|Porsche|Jaguar|Maserati| Honda|Ferrari|Porsche| Land Rover|Chevrolet)', '0'))
#fran1.groupby('franchise_make').count().sort('count',ascending=False).collect()



#torque

tor1=pri2.withColumn('torque', translate('torque', ' ', ''))
tor2 = tor1.withColumn('torque', substring('torque', 1,3))
tor3=tor2.withColumn('torque', regexp_replace('torque', '[^\d{3}]', '0'))
tor4=tor3.withColumn('torque', regexp_replace('torque', '^0[0-4][0-9]', '250'))
tor5=tor4.withColumn('torque', regexp_replace('torque', '^(\d{1,2})$', '250'))
tor6=tor5.withColumn("torque",col("torque").cast("double"))
tor7=tor6.withColumn("torque", col("torque")).na.fill(250)


#days_on_market

day1=tor7.withColumn("daysonmarket",col("daysonmarket").cast("double"))
day2=day1.withColumn("daysonmarket", col("daysonmarket")).na.fill(76)




#engine_displacement

engd1=day2.withColumn('engine_displacement', regexp_replace('engine_displacement', '[^\d{4}]', '0')) 
engd2=engd1.withColumn('engine_displacement', regexp_replace('engine_displacement', '^(\d{5,3500})', '2970'))
engd3=engd2.withColumn('engine_displacement', regexp_replace('engine_displacement', '^0[0-9]*$', '2970'))
engd4=engd3.withColumn('engine_displacement', regexp_replace('engine_displacement', '^[0-9]$', '2970'))
engd5=engd4.withColumn("engine_displacement",col("engine_displacement").cast("double"))
engd6=engd5.withColumn("engine_displacement", col("engine_displacement")).na.fill(2970)



#highway_fuel_economy


hig1=engd6.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '[^\d{3}]', '0')) 
hig2 = hig1.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '(\d{4,3000})', '0')) 
hig3 = hig2.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '^0', '30'))
hig4 = hig3.withColumn('highway_fuel_economy', regexp_replace('highway_fuel_economy', '30[0-9]*$', '30'))
hig5=hig4.withColumn("highway_fuel_economy",col("highway_fuel_economy").cast("double"))
hig6=hig5.withColumn("highway_fuel_economy", col("highway_fuel_economy")).na.fill(30)



#wheel_system

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




#transmission

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




#listing_color

#list_col1=city3.withColumn('listing_color', regexp_replace('listing_color', '(BLACK|WHITE|SILVER|GRAY|BLUE|RED|GREEN|BROWN|YELLOW|GOLD|ORANGE|PURPLE|TEAL|Black (Charcoal)|Gray|jet black|Black (charcoal)|Charcoal Black)', '0'))
#list_col1.groupby('listing_color').count().sort('count',ascending=False).collect() 




#back_legroom

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

#front_legroom

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


#height

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


#length

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

#fuel_tank_volume

fue1=len12.withColumn('fuel_tank_volume', translate('fuel_tank_volume', 'gal', ''))
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

#latitude

#lat1=fue12.withColumn('latitude', regexp_replace('latitude', '[\d\d\.\d\d\d\d*]', '0'))
#lat1.groupby('latitude').count().sort('count',ascending=False).collect()