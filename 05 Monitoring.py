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

# MAGIC %md
# MAGIC ## Compare staging and production model predictions for a user
# MAGIC Changing the wallet address widget to compare different users

# COMMAND ----------

# Read in triplet data
tripletDF = spark.table('g04_db.triple').cache()
tripletDF = tripletDF.withColumnRenamed("addr_id", "user_int_id").withColumnRenamed("tok_id", "token_int_id").withColumnRenamed("bal_usd", "active_holding_usd")
tripletDF = tripletDF.dropna(how="any", subset="active_holding_usd")

# COMMAND ----------

# Predict for a user
from pyspark.sql import functions as F
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

# Print prediction output for staging model
model = mlflow.spark.load_model('models:/'+'ALS'+'/Staging')
predicted_toks = model.transform(tokens_not_held)
    
print('Predicted Tokens:')
toppredictions = predicted_toks.join(tripletDF, 'token_int_id') \
                 .join(tokensDF, tokensDF.id==tripletDF.token_int_id) \
                 .select('name', 'symbol') \
                 .distinct() \
                 .orderBy('prediction', ascending = False)
print(toppredictions.show(5))

# COMMAND ----------

# Print prediction output for production model
model = mlflow.spark.load_model('models:/'+'ALS'+'/Production')
predicted_toks = model.transform(tokens_not_held)
    
print('Predicted Tokens:')
toppredictions = predicted_toks.join(tripletDF, 'token_int_id') \
                 .join(tokensDF, tokensDF.id==tripletDF.token_int_id) \
                 .select('name', 'symbol') \
                 .distinct() \
                 .orderBy('prediction', ascending = False)
print(toppredictions.show(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote staging model to production if it's better

# COMMAND ----------

# Get staging and production models
from mlflow.tracking import MlflowClient
client = MlflowClient()
run_stage = client.get_latest_versions('ALS', ["Staging"])[0]
run_prod = client.get_latest_versions('ALS', ["Production"])[0]

# Gets version of staged model
run_staging=run_stage.version

# Gets version of production model
run_production=run_prod.version

# COMMAND ----------

# If the RSME of staging model is less than that of production, promote staging model and archive production model
if client.get_metric_history(run_id=run_stage.run_id, key='rsme')[0].value < client.get_metric_history(run_id=run_prod.run_id, key='rsme')[0].value:
    filter_string = "name='{}'".format('ALS')
    for mv in client.search_model_versions(filter_string):
         if dict(mv)['current_stage'] == 'Production':
             # Archive the current model in production
             client.transition_model_version_stage(
                name="ALS",
                version=dict(mv)['version'],
                stage="Archived"
            )
    version_fin=int(run_staging)
    client.transition_model_version_stage(
    name='ALS',
    version=version_fin,
    stage='Production'
)

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
