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

# (Take out once active_holding is all set) 
# make active holding positive
#tripletDF = tripletDF.withColumn('active_holding_usd', abs(tripletDF.active_holding_usd))
# Most of the ratings need to not be 0 to train als
#tripletDF = tripletDF.where("active_holding_usd!=0")
tripletDF = tripletDF.where("active_holding_usd>0")
# drop duplicates
#tripletDF = tripletDF.dropDuplicates(['user_int_id', 'token_int_id'])

# COMMAND ----------

# MAGIC %md
# MAGIC Training

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
seed=42
als = ALS(nonnegative = True)

als.setSeed(seed)\
   .setItemCol("token_int_id")\
   .setRatingCol("active_holding_usd")\
   .setUserCol("user_int_id")

# als.setMaxIter(10)\
#    .setSeed(seed)\
#    .setItemCol("token_int_id")\
#    .setRatingCol("active_holding_usd")\
#    .setUserCol("user_int_id")\
#    .setColdStartStrategy("drop")

# COMMAND ----------

# Set params
als.setParams(rank = 2, regParam = 0.2)

# COMMAND ----------

# Fit model (not yet working - needs enough data to work)
model = als.fit(training_df)

# COMMAND ----------

# Test model
predict_df = model.transform(validation_df)

# Remove nan's from prediction
predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))

# Evaluate using RSME
from pyspark.ml.evaluation import RegressionEvaluator
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="active_holding_usd", metricName="rmse") # make sure these column names are correct
error = reg_eval.evaluate(predicted_test_df)

# COMMAND ----------

error

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

# Register model in Staging - copied from MusicRecommenderClass
client = MlflowClient()
model_versions = []

# Transition this model to staging and archive the current staging model if there is one
for mv in client.search_model_versions(f"name='{self.modelName}'"):
    model_versions.append(dict(mv)['version'])
    if dict(mv)['current_stage'] == 'Staging':
    print("Archiving: {}".format(dict(mv)))
    # Archive the currently staged model
    client.transition_model_version_stage(
        name=self.modelName,
        version=dict(mv)['version'],
        stage="Archived"
        )
client.transition_model_version_stage(
    name=self.modelName,
    version=model_versions[0],  # this model (current build)
    stage="Staging"
)

# COMMAND ----------

# Test model in Staging

# COMMAND ----------

# Create a dataframe for best model that will be used for predictions/recommendations

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
