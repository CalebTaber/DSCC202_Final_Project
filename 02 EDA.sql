-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Ethereum Blockchain Data Analysis - <a href=https://github.com/blockchain-etl/ethereum-etl-airflow/tree/master/dags/resources/stages/raw/schemas>Table Schemas</a>
-- MAGIC - **Transactions** - Each block in the blockchain is composed of zero or more transactions. Each transaction has a source address, a target address, an amount of Ether transferred, and an array of input bytes. This table contains a set of all transactions from all blocks, and contains a block identifier to get associated block-specific information associated with each transaction.
-- MAGIC - **Blocks** - The Ethereum blockchain is composed of a series of blocks. This table contains a set of all blocks in the blockchain and their attributes.
-- MAGIC - **Receipts** - the cost of gas for specific transactions.
-- MAGIC - **Traces** - The trace module is for getting a deeper insight into transaction processing. Traces exported using <a href=https://openethereum.github.io/JSONRPC-trace-module.html>Parity trace module</a>
-- MAGIC - **Tokens** - Token data including contract address and symbol information.
-- MAGIC - **Token Transfers** - The most popular type of transaction on the Ethereum blockchain invokes a contract of type ERC20 to perform a transfer operation, moving some number of tokens from one 20-byte address to another 20-byte address. This table contains the subset of those transactions and has further processed and denormalized the data to make it easier to consume for analysis of token transfer events.
-- MAGIC - **Contracts** - Some transactions create smart contracts from their input bytes, and this smart contract is stored at a particular 20-byte address. This table contains a subset of Ethereum addresses that contain contract byte-code, as well as some basic analysis of that byte-code. 
-- MAGIC - **Logs** - Similar to the token_transfers table, the logs table contains data for smart contract events. However, it contains all log data, not only ERC20 token transfers. This table is generally useful for reporting on any logged event type on the Ethereum blockchain.
-- MAGIC 
-- MAGIC ### Rubric for this module
-- MAGIC Answer the quetions listed below.

-- COMMAND ----------

-- MAGIC %run ./includes/utilities

-- COMMAND ----------

-- MAGIC %run ./includes/configuration

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Grab the global variables
-- MAGIC wallet_address,start_date = Utils.create_widgets()
-- MAGIC print(wallet_address,start_date)
-- MAGIC spark.conf.set('wallet.address',wallet_address)
-- MAGIC spark.conf.set('start.date',start_date)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the maximum block number and date of block in the database

-- COMMAND ----------

USE ethereumetl;
SELECT MAX(number), MAX(CAST(CAST(timestamp AS TIMESTAMP) AS DATE)) AS date 
FROM blocks;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: At what block did the first ERC20 transfer happen?

-- COMMAND ----------

USE ethereumetl;
SELECT MIN(block_number)
FROM token_transfers
WHERE token_address IN (SELECT address FROM silver_contracts WHERE is_erc20 = True);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: How many ERC20 compatible contracts are there on the blockchain?

-- COMMAND ----------

USE ethereumetl;

SELECT COUNT(*)
FROM silver_contracts
WHERE is_erc20 = True;

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Q: What percentage of transactions are calls to contracts

-- COMMAND ----------

USE ethereumetl;

SELECT ((SELECT COUNT(*) FROM transactions WHERE to_address IN (SELECT address FROM silver_contracts)) / COUNT(*))*100 AS percentage
FROM transactions;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What are the top 100 tokens based on transfer count?

-- COMMAND ----------

USE ethereumetl;

SELECT COUNT(*) AS frequency, token_address, name
FROM (token_transfers INNER JOIN tokens ON token_address=address)
GROUP BY token_address, name
ORDER BY COUNT(*) DESC LIMIT 100;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What fraction of ERC-20 transfers are sent to new addresses
-- MAGIC (i.e. addresses that have a transfer count of 1 meaning there are no other transfers to this address for this token this is the first)

-- COMMAND ----------

USE ethereumetl;

SELECT (COUNT(to_address)/SUM(count))*100 AS newAddrPercentage
FROM (
  SELECT to_address, COUNT(*) AS count
  FROM token_transfers T INNER JOIN silver_contracts C ON T.token_address = C.address
  WHERE is_erc20 = TRUE
  GROUP BY to_address
);

-- This gets the number of addresses and divides that by the number of transfers
-- This works since every to_address listed in token_transfers must have at least one (first) transfer

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: In what order are transactions included in a block in relation to their gas price?
-- MAGIC - hint: find a block with multiple transactions 

-- COMMAND ----------

USE ethereumetl;

SELECT transaction_index, gas_price
FROM transactions
WHERE block_hash=(SELECT hash
                  FROM blocks
                  WHERE timestamp=(SELECT MAX(timestamp)
                                   FROM blocks))
ORDER BY gas_price DESC;

-- This query shows that transactions with higher gas prices are generally included earlier in the block than transactions with lower gas prices
-- However, there can be slight variations in this rule

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What was the highest transaction throughput in transactions per second?
-- MAGIC hint: assume 15 second block time

-- COMMAND ----------

SELECT MAX(transaction_count)/15 AS MaxThroughput
FROM blocks;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the total Ether volume?
-- MAGIC Note: 1x10^18 wei to 1 eth and value in the transaction table is in wei

-- COMMAND ----------

USE ethereumetl;
SELECT (SUM(value) + SUM(gas)/(1000000000000000000)) AS EtherVolume
FROM transactions;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the total gas used in all transactions?

-- COMMAND ----------

USE ethereumetl;
SELECT SUM(gas) AS TotalGas
FROM transactions;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: Maximum ERC-20 transfers in a single transaction

-- COMMAND ----------

USE ethereumetl;

SELECT MAX(count)
FROM (
      SELECT transaction_hash, COUNT(*) AS count
      FROM token_transfers T INNER JOIN silver_contracts C ON T.token_address = C.address
      WHERE C.is_erc20 = True
      GROUP BY transaction_hash
      );

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: Token balance for any address on any date?

-- COMMAND ----------

USE g04_db;

DROP TABLE IF EXISTS eda_toks_sold;

CREATE TABLE eda_toks_sold(
  token_address STRING,
  amt_sold DECIMAL(38,0)
)
USING DELTA;

INSERT INTO eda_toks_sold
  SELECT token_address, SUM(value)
  FROM ethereumetl.token_transfers
  WHERE from_address = '${wallet.address}'
  GROUP BY token_address;

-- COMMAND ----------

USE g04_db;

DROP TABLE IF EXISTS eda_toks_bought;

CREATE TABLE eda_toks_bought(
  token_address STRING,
  amt_bought DECIMAL(38,0)
)
USING DELTA;

INSERT INTO g04_db.eda_toks_bought
  SELECT token_address, SUM(value)
  FROM ethereumetl.token_transfers
  WHERE to_address = '${wallet.address}'
  GROUP BY token_address;

-- COMMAND ----------

USE g04_db;

SELECT (CASE WHEN B.token_address IS NULL THEN S.token_address
        ELSE B.token_address END) AS token,
       (CASE WHEN S.amt_sold IS NULL THEN B.amt_bought 
             WHEN B.amt_bought IS NULL THEN -S.amt_sold
        ELSE B.amt_bought - S.amt_sold END) AS balance
FROM eda_toks_bought B FULL OUTER JOIN eda_toks_sold S on B.token_address = S.token_address;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Viz the transaction count over time (network use)

-- COMMAND ----------

USE ethereumetl;

SELECT CAST(timestamp AS TIMESTAMP) AS timestamp, transaction_count
FROM blocks;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Viz ERC-20 transfer count over time
-- MAGIC interesting note: https://blog.ins.world/insp-ins-promo-token-mixup-clarified-d67ef20876a3

-- COMMAND ----------

USE ethereumetl;

SELECT CAST(CAST(B.timestamp AS TIMESTAMP) AS DATE) AS date, COUNT(*)
FROM blocks B, token_transfers T
WHERE T.block_number=B.number
GROUP BY B.timestamp;

-- For some reason, the plot does not show up correctly in the output.
-- However, when I click "Plot Options" the preview of the plot looks correct

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Return Success
-- MAGIC dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
