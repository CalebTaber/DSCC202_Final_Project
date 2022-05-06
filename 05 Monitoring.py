# Databricks notebook source
# MAGIC %md
# MAGIC ## Rubric for this module
# MAGIC - Implement a routine to "promote" your model at **Staging** in the registry to **Production** based on a boolean flag that you set in the code.
# MAGIC - Using wallet addresses from your **Staging** and **Production** model test data, compare the recommendations of the two models.

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
# MAGIC ## Your Code Starts Here...

# COMMAND ----------

import random
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from delta.tables import *
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from mlflow.tracking.client import MlflowClient
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %sql
# MAGIC use G04_db;

# COMMAND ----------

df_gold=spark.sql("SELECT * FROM g04_db.prediction_one_user")

# COMMAND ----------

df_gold.display()

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
run_stage = client.get_latest_versions('ALS', ["Staging"])[0]
stat_rmse=client.get_metric_history(run_id=run, key='rsme')[0].value

# COMMAND ----------

run_staging=run_stage.version

# COMMAND ----------

run_prod = client.get_latest_versions('ALS', ["Production"])[0]
prod_rmse=client.get_metric_history(run_id=run, key='rsme')[0].value

# COMMAND ----------

run_production=run_prod.version

# COMMAND ----------

if stat_rmse>prod_rmse:
    version_fin=run_staging
    client.transition_model_version_stage(
    name=model_name,
    version=version_fin,
    stage='Production',
)


# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
