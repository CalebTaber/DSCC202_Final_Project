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
tripletDF = spark.table('g04_db.triple').cache() # we may want to store this as a delta table in BASE_DELTA_PATH and read in from there

# COMMAND ----------

tripletDF.count()

# COMMAND ----------

display(tripletDF)

# COMMAND ----------

tripletDF = tripletDF.withColumnRenamed("addr_id", "user_int_id").withColumnRenamed("tok_id", "token_int_id").withColumnRenamed("bal_usd", "active_holding_usd")

# COMMAND ----------

# Check for nulls
tripletDF.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in tripletDF.columns]).show()

# COMMAND ----------

# Drop na in active holding column
tripletDF = tripletDF.dropna(how="any", subset="active_holding_usd")

# COMMAND ----------

# Ensure rating is cast to float
tripletDF = tripletDF.withColumn("active_holding_usd", tripletDF["active_holding_usd"].cast(DoubleType()))

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
# MAGIC Fit ALS model using grid search

# COMMAND ----------

# Initialize ALS model
from pyspark.ml.recommendation import ALS
seed=42
als = ALS(nonnegative = True)

als.setMaxIter(5)\
   .setSeed(seed)\
   .setItemCol("token_int_id")\
   .setRatingCol("active_holding_usd")\
   .setUserCol("user_int_id")

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="token_int_id", metricName="rmse")

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
grid = ParamGridBuilder() \
  .addGrid(als.maxIter, [10]) \
  .addGrid(als.regParam, [0.15, 0.2, 0.25]) \
  .addGrid(als.rank, [4, 8, 12, 16]) \
  .build()

# COMMAND ----------

cv = CrossValidator(estimator=als, evaluator=reg_eval, estimatorParamMaps=grid, numFolds=3)

# COMMAND ----------

from pyspark.sql import functions as F
tolerance = 0.03
ranks = [2, 4, 8, 12]
regParams = [0.10,0.15, 0.2, 0.25]
errors = [[0]*len(ranks)]*len(regParams)
models = [[0]*len(ranks)]*len(regParams)
err = 0
min_error = float('inf')
best_rank = -1
i = 0
for regParam in regParams:
  j = 0
  for rank in ranks:
    # Set the rank here:
    als.setParams(rank = rank, regParam = regParam)
    # Create the model with these parameters.
    model = als.fit(training_df)
    # Run the model to create a prediction. Predict against the validation_df.
    predict_df = model.transform(validation_df)
 
    # Remove NaN values from prediction (due to SPARK-14489)
    predicted_plays_df = predict_df.filter(predict_df.prediction != float('nan'))
    predicted_plays_df = predicted_plays_df.withColumn("prediction", F.abs(F.round(predicted_plays_df["prediction"],0)))
    # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
    error = reg_eval.evaluate(predicted_plays_df)
    errors[i][j] = error
    models[i][j] = model
    print( 'For rank %s, regularization parameter %s the RMSE is %s' % (rank, regParam, error))
    if error < min_error:
      min_error = error
      best_params = [i,j]
    j += 1
  i += 1
 
als.setRegParam(regParams[best_params[0]])
als.setRank(ranks[best_params[1]])
print( 'The best model was trained with regularization parameter %s' % regParams[best_params[0]])
print( 'The best model was trained with rank %s' % ranks[best_params[1]])
my_model = models[best_params[0]][best_params[1]]

# COMMAND ----------

predicted_plays_df.show(10)

# COMMAND ----------

# Test model
predict_df = model.transform(validation_df)

# Remove nan's from prediction
predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))

# Evaluate using RSME
#from pyspark.ml.evaluation import RegressionEvaluator
#reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="active_holding_usd", metricName="rmse") # make sure these column names are correct
error = reg_eval.evaluate(predicted_test_df)

# COMMAND ----------

display(predict_df)

# COMMAND ----------

error

# COMMAND ----------

# MAGIC %md
# MAGIC ## Push best model to staging

# COMMAND ----------

# Register model in Staging
from mlflow.tracking import MlflowClient
client = MlflowClient()
model_versions = []

# Transition this model to staging and archive the current staging model if there is one
model_run = run_id # insert run id here
filter_string = "run_id='{}'".format(model_run)
for mv in client.search_model_versions(filter_string):
    model_versions.append(dict(mv)['version'])
    if dict(mv)['current_stage'] == 'Staging':
        print("Archiving: {}".format(dict(mv)))
        # Archive the currently staged model
        client.transition_model_version_stage(
            name="ALS model",
            version=dict(mv)['version'],
            stage="Archived"
            )
client.transition_model_version_stage(
    name="ALS",
    version=model_versions[0],  # this model (current build)
    stage="Staging"
)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
model_run = run_id
filter_string = "run_id='{}'".format(model_run)
for mv in client.search_model_versions(filter_string):
    print("name={}; run_id={}; version={}".format(mv.name, mv.run_id, mv.version))

# COMMAND ----------

# Test model in Staging
# Evaluate model
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="active_holding_usd", metricName="rmse")
predict_df = mlflow.spark.load_model('models:/'+"ALS"+'/Staging').stages[0].transform(test_df)
# Remove nan's from prediction
predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))
# Evaluate using RSME
error = reg_eval.evaluate(predicted_test_df)      

# COMMAND ----------

error

# COMMAND ----------

# Predict for a user  
tokensDF = spark.table("g04_db.toks_silver").cache()
addressesDF = spark.table("g04_db.wallet_addrs").cache() 
UserID = addressesDF[addressesDF['addr']==wallet_address].collect()[0][1]
print(UserID)
#UserID = 18330252 
users_tokens = tripletDF.filter(tripletDF.user_int_id == UserID).join(tokensDF, tokensDF.id==tripletDF.token_int_id).select('token_int_id', 'name', 'symbol')                                           
# generate list of tokens held 
tokens_held_list = [] 
# for tok in users_tokens.collect():   
#     tokens_held_list.append(tok['name'])  
print('Tokens user has held:') 
users_tokens.select('name').show()  
    
# generate dataframe of tokens user doesn't have 
tokens_not_held = tripletDF.filter(~ tripletDF['token_int_id'].isin([token['token_int_id'] for token in users_tokens.collect()])).select('token_int_id').withColumn('user_int_id', F.lit(UserID)).distinct()

# COMMAND ----------

# Print prediction output and save top predictions (recommendations) to table in DB
#model = mlflow.spark.load_model('models:/'+'ALS'+'/Staging') ##Uncomment this with name of our best model
predicted_toks = model.transform(tokens_not_held)
    
print('Predicted Tokens:')
toppredictions = predicted_toks.join(tripletDF, 'token_int_id') \
                 .join(tokensDF, tokensDF.id==tripletDF.token_int_id) \
                 .select('name', 'symbol') \
                 .distinct() \
                 .orderBy('prediction', ascending = False)
print(toppredictions.show(5))
spark.sql("""DROP TABLE IF EXISTS g04_db.toprecs_one_user""")
toppredictions.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("g04_db.toprecs_one_user")

# COMMAND ----------

#Create final dataframe and save as table to DB
predicted_toks = predicted_toks.join(addressesDF, predicted_toks.user_int_id == addressesDF.addr_id)
predicted_toks = predicted_toks.join(tokensDF, predicted_toks.token_int_id == tokensDF.id).select("token_int_id", "user_int_id", "prediction", "addr" , "address" ).withColumnRenamed("addr", "user_address").withColumnRenamed("address", "token_address")
spark.sql("""DROP TABLE IF EXISTS g04_db.prediction_one_user""")
predicted_toks.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("g04_db.prediction_one_user")

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
