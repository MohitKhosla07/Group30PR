#!/usr/bin/env python
# coding: utf-8

#CODE CHANGES 

#code changes 2

# In[1]:


get_ipython().system('pip install pyspark')


# In[1]:


from pyspark.sql import SparkSession


# In[2]:


spark = SparkSession         .builder         .appName("LogisticRegressionSummary")         .getOrCreate()


# In[3]:


from pyspark.ml.classification import LogisticRegression


# In[20]:


#file_location = "E:\BIG DATA WORKSPACE\Mini Project\Data.csv"
file_type = "csv"

# CSV options
infer_schema = True
first_row_is_header = False
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
rawstrokeDF = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load("E:/Mini Project/data.csv")


# In[ ]:


rawstrokeDF.show(5, False)


# In[ ]:


from pyspark.sql.functions import isnull, when, count, col
rawstrokeDF.select([count(when(isnull(c), c)).alias(c) for c in rawstrokeDF.columns]).show()


# In[7]:


rawstrokeDF.printSchema()


# In[8]:


rawstrokeDF.count()


# In[9]:


from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

# df2 = rawstrokeDF.withColumn("_c0",rawstrokeDF._c0.cast('double'))


# In[10]:


intcols = ['_c0','_c5','_c13','_c14','_c15','_c16','_c17','_c18','_c19','_c21','_c22','_c23','_c24','_c26','_c27','_c28','_c29','_c30','_c31','_c32','_c33','_c34','_c35','_c36']
for col_name in intcols:
    rawstrokeDF=rawstrokeDF.withColumn(col_name,col(col_name).cast('double'))


# In[11]:


rawstrokeDF.printSchema()


# In[12]:


type(rawstrokeDF)


# In[13]:


get_ipython().system('pip install pyArrow')


# In[14]:


df1=rawstrokeDF.to_pandas_on_spark()


# In[15]:


df1.columns = ["Target","NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
             "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","DAYS_BIRTH","DAYS_EMPLOYED","OWN_CAR_AGE","FLAG_MOBIL",
             "FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_CONT_MOBILE","FLAG_PHONE","OCCUPATION_TYPE","CNT_FAM_MEMBERS","REGION_RATING_CLIENT",
             "REGION_RATING_CLIENT_W_CITY","REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","ORGANIZATION_TYPE","FLAG_DOCUMENT_2",
             "FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6","FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9",
             "FLAG_DOCUMENT_10"]


# In[16]:


df1.head(5)


# In[17]:


type(df1)


# In[18]:


rawstrokeDF = df1.to_spark()


# In[19]:


type(rawstrokeDF)


# In[20]:


rawstrokeDF.printSchema()


# In[21]:


#from pyspark.ml.feature import OneHotEncoder,StringIndexer


# In[22]:


# #Create StringIndexer Object 
# SI_NAME_CONTRACT_TYPE = StringIndexer(inputCol="NAME_CONTRACT_TYPE",outputCol="NAME_CONTRACT_TYPE_Index")
# SI_CODE_GENDER = StringIndexer(inputCol="CODE_GENDER",outputCol="CODE_GENDER_Index")
# SI_FLAG_OWN_CAR= StringIndexer(inputCol="FLAG_OWN_CAR",outputCol="FLAG_OWN_CAR_Index")
# SI_FLAG_OWN_REALTY = StringIndexer(inputCol="FLAG_OWN_REALTY",outputCol="FLAG_OWN_REALTY_Index")
# SI_NAME_INCOME_TYPE= StringIndexer(inputCol="NAME_INCOME_TYPE",outputCol="NAME_INCOME_TYPE_Index")
# SI_NAME_EDUCATION_TYPE= StringIndexer(inputCol="NAME_EDUCATION_TYPE",outputCol="NAME_EDUCATION_TYPE_Index")
# SI_NAME_FAMILY_STATUS = StringIndexer(inputCol="NAME_FAMILY_STATUS",outputCol="NAME_FAMILY_STATUS_Index")
# SI_NAME_HOUSING_TYPE = StringIndexer(inputCol="NAME_HOUSING_TYPE",outputCol="NAME_HOUSING_TYPE_Index")
# SI_REG_REGION_NOT_LIVE_REGION = StringIndexer(inputCol="REG_REGION_NOT_LIVE_REGION",outputCol="REG_REGION_NOT_LIVE_REGION_Index")


# In[23]:


# #transform the data 
# rawstrokeDF = SI_NAME_CONTRACT_TYPE.fit(rawstrokeDF).transform(rawstrokeDF)
# rawstrokeDF = SI_CODE_GENDER.fit(rawstrokeDF).transform(rawstrokeDF)
# rawstrokeDF = SI_FLAG_OWN_CAR.fit(rawstrokeDF).transform(rawstrokeDF)
# rawstrokeDF = SI_FLAG_OWN_REALTY.fit(rawstrokeDF).transform(rawstrokeDF)
# rawstrokeDF = SI_NAME_INCOME_TYPE.fit(rawstrokeDF).transform(rawstrokeDF)
# rawstrokeDF = SI_NAME_EDUCATION_TYPE.fit(rawstrokeDF).transform(rawstrokeDF)
# rawstrokeDF = SI_NAME_FAMILY_STATUS.fit(rawstrokeDF).transform(rawstrokeDF)
# rawstrokeDF = SI_NAME_HOUSING_TYPE.fit(rawstrokeDF).transform(rawstrokeDF)
# rawstrokeDF = SI_REG_REGION_NOT_LIVE_REGION.fit(rawstrokeDF).transform(rawstrokeDF)


# In[24]:


#OHE = OneHotEncoder(inputCols =["NAME_CONTRACT_TYPE_Index","CODE_GENDER_Index","FLAG_OWN_CAR_Index","FLAG_OWN_REALTY_Index","NAME_INCOME_TYPE_Index","NAME_EDUCATION_TYPE_Index","NAME_FAMILY_STATUS_Index","NAME_HOUSING_TYPE_Index","REG_REGION_NOT_LIVE_REGION_Index"], outputCols=["NAME_CONTRACT_TYPE_OHE","CODE_GENDER_OHE","FLAG_OWN_CAR_OHE","FLAG_OWN_REALTY_OHE","NAME_INCOME_TYPE_OHE","NAME_EDUCATION_TYPE_OHE","NAME_FAMILY_STATUS_OHE","NAME_HOUSING_TYPE_OHE","REG_REGION_NOT_LIVE_REGION_OHE"])


# In[25]:


#rawstrokeDF= OHE.fit(rawstrokeDF).transform(rawstrokeDF)


# In[26]:


#rawstrokeDF.show(5,False)


# In[27]:


#view and transform data 
#rawstrokeDF.select("NAME_CONTRACT_TYPE_Index","CODE_GENDER_Index","FLAG_OWN_CAR_Index","FLAG_OWN_REALTY_Index","NAME_INCOME_TYPE_Index","NAME_EDUCATION_TYPE_Index","NAME_FAMILY_STATUS_Index","NAME_HOUSING_TYPE_Index","REG_REGION_NOT_LIVE_REGION_Index","NAME_CONTRACT_TYPE_OHE","CODE_GENDER_OHE","FLAG_OWN_CAR_OHE","FLAG_OWN_REALTY_OHE","NAME_INCOME_TYPE_OHE","NAME_EDUCATION_TYPE_OHE","NAME_FAMILY_STATUS_OHE","NAME_HOUSING_TYPE_OHE","REG_REGION_NOT_LIVE_REGION_OHE").show(10)


# In[28]:


#from pyspark.ml.feature import VectorIndexer


# In[29]:


#features_col =["Target","CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DAYS_BIRTH","DAYS_EMPLOYED","OWN_CAR_AGE","FLAG_MOBIL","FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_CONT_MOBILE","FLAG_PHONE","OCCUPATION_TYPE","CNT_FAM_MEMBERS","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","REG_REGION_NOT_WORK_REGION","ORGANIZATION_TYPE","FLAG_DOCUMENT_2","FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6","FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9","FLAG_DOCUMENT_10", "NAME_CONTRACT_TYPE_OHE","CODE_GENDER_OHE","FLAG_OWN_CAR_OHE","FLAG_OWN_REALTY_OHE","NAME_INCOME_TYPE_OHE","NAME_EDUCATION_TYPE_OHE","NAME_FAMILY_STATUS_OHE","NAME_HOUSING_TYPE_OHE","REG_REGION_NOT_LIVE_REGION_OHE"]


# In[30]:


cat_features=["NAME_CONTRACT_TYPE","CODE_GENDER", "FLAG_OWN_CAR","FLAG_OWN_REALTY", "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE", "REG_REGION_NOT_LIVE_REGION"]


# In[31]:


len(cat_features)


# In[32]:


cont_features=["Target","CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DAYS_BIRTH","DAYS_EMPLOYED","OWN_CAR_AGE","FLAG_MOBIL","FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_CONT_MOBILE","FLAG_PHONE","OCCUPATION_TYPE","CNT_FAM_MEMBERS","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","REG_REGION_NOT_WORK_REGION","ORGANIZATION_TYPE","FLAG_DOCUMENT_2","FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5","FLAG_DOCUMENT_6","FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9","FLAG_DOCUMENT_10"]


# In[33]:


len(cont_features)


# In[34]:


from pyspark.sql.functions import col


# In[35]:


from pyspark.ml.feature import OneHotEncoder,StringIndexer,VectorAssembler


# In[36]:


from pyspark.ml.feature import OneHotEncoder,StringIndexer,VectorAssembler

# defining an empty list to hold transforming stages
# to prepare pipelines
stages=[]


# Encoding categorical features
for catcol in cat_features:
    indexer=StringIndexer(inputCol=catcol,outputCol=catcol+'_index').setHandleInvalid("keep")
    encoder=OneHotEncoder(inputCols=[indexer.getOutputCol()],outputCols=[catcol+"_enc"])
    stages+=[indexer,encoder]


# In[37]:


assemblerInputs=[col+"_enc" for col in cat_features]+cont_features
assembler=VectorAssembler(inputCols=assemblerInputs,outputCol="features")
stages+=[assembler]


# In[38]:


# Scaling the features vector
from pyspark.ml.feature import MinMaxScaler
scaler = MinMaxScaler().setInputCol("features").setOutputCol("scaled_features")
stages+=[scaler]


# In[39]:


# Building a spark ml pipeline to transform the data
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=stages)

# Fit the pipeline to training documents.
finalDF = pipeline.fit(rawstrokeDF).transform(rawstrokeDF)


# In[40]:


finalDF.printSchema()


# In[41]:


trainDF, testDF =  finalDF.randomSplit([0.7,0.3], seed = 2020)

# print the count of observations in each set
print("Observations in training set = ", trainDF.count())
print("Observations in testing set = ", testDF.count())


# In[42]:


from pyspark.ml.classification import LogisticRegression

# Build the LogisticRegression object 'lr' by setting the required parameters
lr = LogisticRegression(featuresCol="features", labelCol="Target",maxIter= 10,regParam=0.3, elasticNetParam=0.8)

# fit the LogisticRegression object on the training data
lrmodel = lr.fit(trainDF)


# In[43]:


predictonDF = lrmodel.transform(testDF)


# In[44]:


predictonDF.select("Target","rawPrediction", "probability", "prediction").show(10,False)


# In[45]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Build the BinaryClassificationEvaluator object 'evaluator'
evaluator = BinaryClassificationEvaluator()

# Calculate the accracy and print its value
accuracy = predictonDF.filter(predictonDF.Target == predictonDF.prediction).count()/float(predictonDF.count())
print("Accuracy = ", accuracy)

# evaluate(predictiondataframe) gets area under the ROC curve
#print('Area under the ROC curve = ', evaluator.evaluate(predictonDF))


# In[46]:


# Create model summary object
lrmodelSummary = lrmodel.summary

# Print the following metrics one by one: 
# 1. Accuracy
# Accuracy is a model summary parameter
print("Accuracy = ", lrmodelSummary.accuracy)
# 2. Area under the ROC curve
# Area under the ROC curve is a model summary parameter
print("Area under the ROC curve = ", lrmodelSummary.areaUnderROC)
# 3. Precision (Positive Predictive Value)
# Precision is a model summary parameter
print("Precision = ", lrmodelSummary.weightedPrecision)
# 4. Recall (True Positive Rate)
# Recall is a model summary parameter
print("Recall = ", lrmodelSummary.weightedRecall)
# 5. F1 Score (F-measure)
# F1 Score is a model summary method
print("F1 Score = ", lrmodelSummary.weightedFMeasure())


# In[47]:


pip install kafka-python


# In[49]:


get_ipython().system('pip install zookeeper')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




