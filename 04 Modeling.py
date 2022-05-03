# Databricks notebook source
# MAGIC %md
# MAGIC ## Rubric for this module
# MAGIC - Using the silver delta table(s) that were setup by your ETL module train and validate your token recommendation engine. Split, Fit, Score, Save
# MAGIC - Log all experiments using mlflow
# MAGIC - capture model parameters, signature, training/test metrics and artifacts
# MAGIC - Tune hyperparameters using an appropriate scaling mechanism for spark.  [Hyperopt/Spark Trials ](https://docs.databricks.com/_static/notebooks/hyperopt-spark-ml.html)
# MAGIC - Register your best model from the training run at **Staging**.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Your Code starts here...

# COMMAND ----------

# Create/set experiment for group
MY_EXPERIMENT = "/Users/slogan6@ur.rochester.edu/Final_Project_Experiment"
mlflow.set_experiment(MY_EXPERIMENT)

# COMMAND ----------

# Check experiment
experiment = mlflow.get_experiment_by_name("/Users/slogan6@ur.rochester.edu/Final_Project_Experiment")
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

# COMMAND ----------

# Read in triplet data - wallets is currently a subset of all data
tripletDF = spark.table('g04_db.wallets').cache() # we may want to store this as a delta table in BASE_DELTA_PATH and read in from there

# COMMAND ----------

display(tripletDF)

# COMMAND ----------

# Ensure/fix wallet_hash and token_address are ints
# We'll need silver tables for converting wallet hashes and token addresses to their respective int so we can get it back at the end
# For now I'll just do it randomly
from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number
from pyspark.sql.functions import sequence
from pyspark.sql.window import Window

# Get list of unique users
unique_users = tripletDF.select('wallet_hash').distinct().collect()
# Create dataframe for wallet hash and unique integer user id
spark = SparkSession.builder.getOrCreate()
usersDF = spark.createDataFrame(unique_users)
usersDF = usersDF.select("*").withColumn("user_int_id", row_number().over(Window.partitionBy().orderBy(usersDF['wallet_hash'])))
# Join the integer user id to triplet table
tripletDF = tripletDF.join(usersDF, "wallet_hash", "inner")

# Get list of unique tokens
unique_tokens = tripletDF.select('token_address').distinct().collect()
# Create dataframe for token address and unique integer token id
spark = SparkSession.builder.getOrCreate()
tokensDF = spark.createDataFrame(unique_tokens)
tokensDF = tokensDF.select("*").withColumn("token_int_id", row_number().over(Window.partitionBy().orderBy(tokensDF['token_address'])))
# Join the integer token id to triplet table
tripletDF = tripletDF.join(tokensDF, "token_address", "inner")

# COMMAND ----------

display(tripletDF)

# COMMAND ----------

# Drop na in active holding column
tripletDF = tripletDF.dropna(how="any", subset="active_holding_usd")

# COMMAND ----------

display(tripletDF)

# COMMAND ----------

# Ensure rating is cast to float
tripletDF = tripletDF.withColumn("active_holding_usd", tripletDF["active_holding_usd"].cast(DoubleType()))

# COMMAND ----------

# (Take out once active_holding is all set) 
# make active holding positive
tripletDF = tripletDF.withColumn('active_holding_usd', abs(tripletDF.active_holding_usd))
# Most of the ratings need to not be 0 to train als
tripletDF = tripletDF.where("active_holding_usd!=0")
# drop duplicates
tripletDF = tripletDF.dropDuplicates(['user_int_id', 'token_int_id'])

# COMMAND ----------

# Split data
seed = 42
(split_60_df, split_a_20_df, split_b_20_df) = tripletDF.randomSplit([0.6, 0.2, 0.2], seed = seed)
# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC Trying to fit a single ALS model first

# COMMAND ----------

# Initialize ALS model
from pyspark.ml.recommendation import ALS
als = ALS()
als.setMaxIter(5)\
   .setSeed(seed)\
   .setItemCol("token_int_id")\
   .setRatingCol("active_holding_usd")\
   .setUserCol("user_int_id")\
   .setColdStartStrategy("drop")

# COMMAND ----------

# Set params
als.setParams(rank = 2, regParam = 0.1)

# COMMAND ----------

# Fit model (not yet working - needs enough data to work)
model = als.fit(training_df)

# COMMAND ----------

# Test model
predict_df = model.transform(validation_df)

# Remove nan's from prediction
predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))

# Evaluate using RSME
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="active_holding_usd", metricName="rmse") # make sure these column names are correct
error = reg_eval.evaluate(predicted_test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Fit models using Hyperopt

# COMMAND ----------

# Function to train ALS (needs to return model and score)
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def train_ALS(regParam, rank):
    with mlflow.start_run(nested=True):
        # insert above code once working
        # Instantiate model
        als = ALS()
        als.setMaxIter(5)\
           .setSeed(seed)\
           .setItemCol("token_int_id")\
           .setRatingCol("active_holding_usd")\
           .setUserCol("user_int_id")\
           .setColdStartStrategy("drop")
        als.setParams(rank = rank, regParam = regParam)
        
        # Fit model to training data
        model = als.fit(training_df)
        
        # Evaluate model
        predict_df = model.transform(validation_df)
        # Remove nan's from prediction
        predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))
        # Evaluate using RSME
        reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="active_holding_usd", metricName="rmse")
        error = reg_eval.evaluate(predicted_test_df)      
        
        # Log evaluation metric
        mlflow.log_metric("rsme", error)
        
    return als, error

# COMMAND ----------

# Test that train_ALS() function works

# COMMAND ----------

# Hyperopt for hyperparameter tuning
from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

def train_with_hyperopt(params):
    regParam = float(params['regParam'])
    rank = int(params['rank'])
    model, rsme = train_ALS(regParam, rank)
    return {'rsme': rsme, 'status': STATUS_OK}

# COMMAND ----------

# Define hyperopt search space
space = {'regParam': hp.uniform('regParam', 0.1, 0.3),
        'rank': hp.uniform('rank', 4, 16)}

# COMMAND ----------

# Run hyperopt with mlflow - does hyperopt log params and some metrics?
# took out trials=spark_trials because https://docs.databricks.com/_static/notebooks/hyperopt-spark-ml.html says to
with mlflow.start_run():
    best_hyperparam = fmin(fn=train_with_hyperopt, 
                         space=space, 
                         algo=tpe.suggest, 
                         max_evals=10)

# COMMAND ----------

# Use hyperopt to get best params for model
import hyperopt

print(hyperopt.space_eval(space, best_hyperparam))
print(best_hyperparam)

# COMMAND ----------

# If autologging is on, end run with this command
# mlflow.end_run()

# COMMAND ----------

# Register model in Staging

# COMMAND ----------

# Test model in Staging

# COMMAND ----------

# Create a dataframe for best model that will be used for predictions/recommendations

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
