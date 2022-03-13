#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Cleaning

# 'owner_count'

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


sr1=mil10.withColumn('seller_rating', translate('seller_rating', ' ', ''))
sr2=sr1.withColumn('seller_rating', regexp_replace('seller_rating', '[^(\d\d+)]', '0'))
sr3=sr2.withColumn('seller_rating', regexp_replace('seller_rating', '(\s+\d+\s+)$', '')) 
sr4=sr3.withColumn('seller_rating', regexp_replace('seller_rating', '[^\d{1}]', '0'))
sr5=sr4.withColumn('seller_rating', regexp_replace('seller_rating', '^(\d{2,2500})', '4'))
sr6=sr5.withColumn('seller_rating', regexp_replace('seller_rating', '[^1-5$]', '4'))
sr7=sr6.withColumn('seller_rating',col('seller_rating').cast('double'))
sr8=sr7.withColumn('seller_rating', col('seller_rating')).na.fill(4)


# In[ ]:


#fran1=list_col1.withColumn('franchise_make', regexp_replace('franchise_make', '(Ford|Chevrolet|Honda|Toyota|Jeep|Hyundai|Nissan|Kia|Subaru|Buick|Volkswagen|Mazda|BMW|GMC|Mercedes-Benz|Volvo|Audi|Cadillac|Dodge|Mitsubishi|Land Rover|Porsche|Jaguar|Maserati| Honda|Ferrari|Porsche| Land Rover|Chevrolet)', '0'))
#fran1.groupby('franchise_make').count().sort('count',ascending=False).collect()


# In[ ]:




