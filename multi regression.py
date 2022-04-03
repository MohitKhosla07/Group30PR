#Multiple linear regression



lr1 = LinearRegression(featuresCol="scaledfeatures", labelCol="price")


pred_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="r2")


from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator


paramGrid = ParamGridBuilder() \
    .addGrid(lr1.elasticNetParam, [0.1, 0.2, 0.2]) \
    .addGrid(lr1.regParam, [0.1, 0.01]) \
    .build()

crossval = CrossValidator(estimator=lr1,
                          estimatorParamMaps=paramGrid,
                          evaluator=pred_evaluator,
                          numFolds=5)


cvModel1 = crossval.fit(trainDF)


prediction1 = cvModel.transform(testDF)


prediction1.columns


prediction1.select("price", "prediction").show(20)