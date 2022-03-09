import time
import os
import random
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator

# Set path
os.chdir(os.path.join('/home','hduser', '\xe6\xa1\x8c\xe9\x9d\xa2'))
spark = SparkSession.builder.master('local[*]').getOrCreate()
spark = SparkSession.builder.appName('Recommendations').getOrCreate()

# Read csv file
movie_rating = spark.read.csv('./movieRating.csv',header=True)
movie_rating.show()

# Transform the data
movie_rating = movie_rating.withColumn('UserID', movie_rating['UserID'].cast('float'))
movie_rating = movie_rating.withColumn('MovieID', movie_rating['MovieID'].cast('float'))
movie_rating = movie_rating.withColumn('Rating', movie_rating['Rating'].cast('float'))

# Split the dataset
(train, test) = movie_rating.randomSplit([0.8, 0.2], seed=2020)

# Set the model
als = ALS(
    userCol="UserID", 
    itemCol="MovieID",
    ratingCol="Rating", 
    nonnegative=True, 
    implicitPrefs=False,
)

# Search the best hyperparameter
param_grid = ParamGridBuilder().\
             addGrid(als.rank, [12]).\
             addGrid(als.regParam, [0.17]).build()

# Set the evaluation metrics
evaluator = RegressionEvaluator(
           metricName="mae",
           labelCol="Rating", 
           predictionCol="prediction") 
print("Num models to be tested: ", len(param_grid))

tvs = TrainValidationSplit(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=evaluator)

model = tvs.fit(train)
best_model = model.bestModel

# Predict the test data
test_predictions = best_model.transform(test)
test_predictions = test_predictions.dropna()
error_term = evaluator.evaluate(test_predictions)
print(error_term)

print("**Best Model**")
print("Rank:", best_model._java_obj.parent().getRank())
print("MaxIter:", best_model._java_obj.parent().getMaxIter())
print("RegParam:", best_model._java_obj.parent().getRegParam())
test_predictions.sort('UserID', 'Rating').show()
