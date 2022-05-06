# Databricks notebook source
# MAGIC %md
# MAGIC ## Ethereum Blockchain Data Analysis - <a href=https://github.com/blockchain-etl/ethereum-etl-airflow/tree/master/dags/resources/stages/raw/schemas>Table Schemas</a>
# MAGIC - **Transactions** - Each block in the blockchain is composed of zero or more transactions. Each transaction has a source address, a target address, an amount of Ether transferred, and an array of input bytes. This table contains a set of all transactions from all blocks, and contains a block identifier to get associated block-specific information associated with each transaction.
# MAGIC - **Blocks** - The Ethereum blockchain is composed of a series of blocks. This table contains a set of all blocks in the blockchain and their attributes.
# MAGIC - **Receipts** - the cost of gas for specific transactions.
# MAGIC - **Traces** - The trace module is for getting a deeper insight into transaction processing. Traces exported using <a href=https://openethereum.github.io/JSONRPC-trace-module.html>Parity trace module</a>
# MAGIC - **Tokens** - Token data including contract address and symbol information.
# MAGIC - **Token Transfers** - The most popular type of transaction on the Ethereum blockchain invokes a contract of type ERC20 to perform a transfer operation, moving some number of tokens from one 20-byte address to another 20-byte address. This table contains the subset of those transactions and has further processed and denormalized the data to make it easier to consume for analysis of token transfer events.
# MAGIC - **Contracts** - Some transactions create smart contracts from their input bytes, and this smart contract is stored at a particular 20-byte address. This table contains a subset of Ethereum addresses that contain contract byte-code, as well as some basic analysis of that byte-code. 
# MAGIC - **Logs** - Similar to the token_transfers table, the logs table contains data for smart contract events. However, it contains all log data, not only ERC20 token transfers. This table is generally useful for reporting on any logged event type on the Ethereum blockchain.
# MAGIC 
# MAGIC In Addition, there is a price feed that changes daily (noon) that is in the **token_prices_usd** table
# MAGIC 
# MAGIC ### Rubric for this module
# MAGIC - Transform the needed information in ethereumetl database into the silver delta table needed by your modeling module
# MAGIC - Clearly document using the notation from [lecture](https://learn-us-east-1-prod-fleet02-xythos.content.blackboardcdn.com/5fdd9eaf5f408/8720758?X-Blackboard-Expiration=1650142800000&X-Blackboard-Signature=h%2FZwerNOQMWwPxvtdvr%2FmnTtTlgRvYSRhrDqlEhPS1w%3D&X-Blackboard-Client-Id=152571&response-cache-control=private%2C%20max-age%3D21600&response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%27Delta%2520Lake%2520Hands%2520On%2520-%2520Introduction%2520Lecture%25204.pdf&response-content-type=application%2Fpdf&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAAaCXVzLWVhc3QtMSJHMEUCIQDEC48E90xPbpKjvru3nmnTlrRjfSYLpm0weWYSe6yIwwIgJb5RG3yM29XgiM%2BP1fKh%2Bi88nvYD9kJNoBNtbPHvNfAqgwQIqP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw2MzU1Njc5MjQxODMiDM%2BMXZJ%2BnzG25TzIYCrXAznC%2BAwJP2ee6jaZYITTq07VKW61Y%2Fn10a6V%2FntRiWEXW7LLNftH37h8L5XBsIueV4F4AhT%2Fv6FVmxJwf0KUAJ6Z1LTVpQ0HSIbvwsLHm6Ld8kB6nCm4Ea9hveD9FuaZrgqWEyJgSX7O0bKIof%2FPihEy5xp3D329FR3tpue8vfPfHId2WFTiESp0z0XCu6alefOcw5rxYtI%2Bz9s%2FaJ9OI%2BCVVQ6oBS8tqW7bA7hVbe2ILu0HLJXA2rUSJ0lCF%2B052ScNT7zSV%2FB3s%2FViRS2l1OuThecnoaAJzATzBsm7SpEKmbuJkTLNRN0JD4Y8YrzrB8Ezn%2F5elllu5OsAlg4JasuAh5pPq42BY7VSKL9VK6PxHZ5%2BPQjcoW0ccBdR%2Bvhva13cdFDzW193jAaE1fzn61KW7dKdjva%2BtFYUy6vGlvY4XwTlrUbtSoGE3Gr9cdyCnWM5RMoU0NSqwkucNS%2F6RHZzGlItKf0iPIZXT3zWdUZxhcGuX%2FIIA3DR72srAJznDKj%2FINdUZ2s8p2N2u8UMGW7PiwamRKHtE1q7KDKj0RZfHsIwRCr4ZCIGASw3iQ%2FDuGrHapdJizHvrFMvjbT4ilCquhz4FnS5oSVqpr0TZvDvlGgUGdUI4DCdvOuSBjqlAVCEvFuQakCILbJ6w8WStnBx1BDSsbowIYaGgH0RGc%2B1ukFS4op7aqVyLdK5m6ywLfoFGwtYa5G1P6f3wvVEJO3vyUV16m0QjrFSdaD3Pd49H2yB4SFVu9fgHpdarvXm06kgvX10IfwxTfmYn%2FhTMus0bpXRAswklk2fxJeWNlQF%2FqxEmgQ6j4X6Q8blSAnUD1E8h%2FBMeSz%2F5ycm7aZnkN6h0xkkqQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220416T150000Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21600&X-Amz-Credential=ASIAZH6WM4PLXLBTPKO4%2F20220416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=321103582bd509ccadb1ed33d679da5ca312f19bcf887b7d63fbbb03babae64c) how your pipeline is structured.
# MAGIC - Your pipeline should be immutable
# MAGIC - Use the starting date widget to limit how much of the historic data in ethereumetl database that your pipeline processes.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)
spark.conf.set('wallet.address',wallet_address)
spark.conf.set('start.date',start_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## YOUR SOLUTION STARTS HERE...

# COMMAND ----------

# MAGIC %run ./includes/ETLFunctions

# COMMAND ----------

# CALEB
# Strips down the tokens table to only ERC20 tokens. Also, adds pricing information
# Only tracks tokens included in the token_prices_usd table since tokens without pricing info are not of interest to us
# Only needs to be run once per day so that the token prices are up-to-date

spark.sql("""DROP TABLE IF EXISTS g04_db.toks_silver""")

tokens_silver = create_tokens_silver()

(
  tokens_silver.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("address")
    .saveAsTable("g04_db.toks_silver")
)

# COMMAND ----------

# CALEB
# Strips down the token_transfers table to a more manageable set of useful attributes
# Also removes transfers that involve tokens not stored in the tokens_silver table (see above command)

# Only the token ID -- NOT THE TOKEN ADDRESS -- is stored in this table
# This helps save space and (I hope) speeds up table manipulation

spark.sql("""DROP TABLE IF EXISTS g04_db.tt_silver""")

tt_silver = create_tt_silver()

(tt_silver.write
          .format("delta")
          .mode("overwrite")
          .partitionBy("id")
          .saveAsTable('g04_db.tt_silver'))

spark.sql("""OPTIMIZE g04_db.tt_silver ZORDER BY (to_address)""")

# COMMAND ----------

# CALEB
# Fills the abridged token transfers table given a start date specified in the widget
# Just creates a subset of the token_transfers_silver table

tt_silver = spark.table('g04_db.tt_silver')

tt_silver_abridged = (
                      tt_silver.select('*')
                               .where(tt_silver.timestamp > '2021-01-01T00:00:00.000+000')
                     ).drop('timestamp')

(tt_silver_abridged.write
                   .format("delta")
                   .mode("overwrite")
                   .saveAsTable("g04_db.tt_silver_abridged"))

# COMMAND ----------

# Using the tt_silver & toks_silver tables, finds all addresses that bought coins and groups their total purchases

spark.sql("""DROP TABLE IF EXISTS g04_db.bought""")

bought = create_bought()

bought.write\
      .format("delta")\
      .mode("overwrite")\
      .partitionBy("id")\
      .saveAsTable("g04_db.bought")

spark.sql("""OPTIMIZE g04_db.bought ZORDER BY (to_addr)""")

# COMMAND ----------

# Same as bought but centered around the from_address showing total sales per coin

spark.sql("""DROP TABLE IF EXISTS g04_db.sold""")

sold = create_sold()

sold.write\
      .format("delta")\
      .mode("overwrite")\
      .partitionBy("id")\
      .saveAsTable("g04_db.sold")
    
spark.sql("""OPTIMIZE g04_db.sold ZORDER BY (from_addr)""")

# COMMAND ----------

#Combines the bought and sold tables creating an active holding wallet

spark.sql("""DROP TABLE IF EXISTS g04_db.with_bal""")

with_bal = create_with_bal()

with_bal.write\
        .format("delta")\
        .mode("overwrite")\
        .partitionBy("id")\
        .saveAsTable("g04_db.with_bal")

spark.sql("""OPTIMIZE g04_db.with_bal ZORDER BY (addr)""")

# COMMAND ----------

#Converts the raw coin count from the wallet created above to a dollar value
spark.sql("""DROP TABLE IF EXISTS g04_db.with_usd""")

with_usd = create_with_usd()

with_usd.write\
        .format("delta")\
        .mode("overwrite")\
        .partitionBy("id")\
        .saveAsTable("g04_db.with_usd")

spark.sql("""OPTIMIZE g04_db.with_usd ZORDER BY (addr)""")

# COMMAND ----------

#Finds the unique wallets with a row number integer ID

spark.sql("""DROP TABLE IF EXISTS g04_db.wallet_addrs""")

wallet_addrs = create_wallet_addrs()

wallet_addrs.write\
            .format("delta")\
            .mode("overwrite")\
            .saveAsTable("g04_db.wallet_addrs")


# COMMAND ----------

#Finalizes the wallet construction creating a triple with a wallet ID int, token ID int and a value based on USD

spark.sql("""DROP TABLE IF EXISTS g04_db.triple""")

triple = create_triple()

triple.write\
      .format("delta")\
      .mode("overwrite")\
      .saveAsTable("g04_db.triple")

# COMMAND ----------

#triOrig = spark.table('g04_db.triple')
triple.count()

# COMMAND ----------

triple.count()

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
