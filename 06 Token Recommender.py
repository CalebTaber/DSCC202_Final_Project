# Databricks notebook source
# MAGIC %md
# MAGIC ## Token Recommendation
# MAGIC <table border=0>
# MAGIC   <tr><td><img src='https://data-science-at-scale.s3.amazonaws.com/images/rec-application.png'></td>
# MAGIC     <td>Your application should allow a specific wallet address to be entered via a widget in your application notebook.  Each time a new wallet address is entered, a new recommendation of the top tokens for consideration should be made. <br> **Bonus** (3 points): include links to the Token contract on the blockchain or etherscan.io for further investigation.</td></tr>
# MAGIC   </table>

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
# MAGIC ## Your code starts here...

# COMMAND ----------

##Currently Everything is selected to compare the name and symbol, for the actual application, comment out the select all and uncomment the top line. 
rec = spark.sql("""
-- SELECT g04_db.toprecs_one_user.name, g04_db.toprecs_one_user.symbol, image, contract_address
SELECT *
FROM g04_db.toprecs_one_user
LEFT JOIN ethereumetl.token_prices_usd
ON UPPER(g04_db.toprecs_one_user.symbol) = UPPER(ethereumetl.token_prices_usd.symbol) 
-- AND UPPER(g04_db.toprecs_one_user.name) = UPPER(ethereumetl.token_prices_usd.name) 
LIMIT 10
""").toPandas()

rec
##uncomment to see the application
# results = ""
# if(rec.empty):
#     displayHTML("""<h2>Error: Recommendation Empty</h2>""")
# else:
#     for index, row in rec.iterrows():
#         results += """
#         <tr>
#           <td style="padding:.5rem"><img src="{2}"></td>          
#           <td style="padding:.5rem">{0} ({1})</td>          
#           <td style="padding:.5rem"> <a href="https://etherscan.io/address/{3}"> <img src="https://freeiconshop.com/wp-content/uploads/edd/link-closed-flat.png" width="25" height="25"> </a> </td>
#         </tr>
#         """.format(row["name"], row["symbol"], row["image"] , row["contract_address"])
    
#     displayHTML("""
#         <h2>Recommend Tokens for user address:</h2>
#         <table style="padding:.5rem, border:0">
#         {0}
#         </table>
#     """.format(results))

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


