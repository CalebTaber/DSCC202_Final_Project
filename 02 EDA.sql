-- Databricks notebook source
USE g04_db;

DROP TABLE IF EXISTS tokens_silver;

CREATE TABLE tokens_silver USING DELTA
(
  address STRING,
  name STRING,
  price_usd DOUBLE
);

INSERT INTO tokens_silver
  SELECT contract_address, name, price_usd
  FROM ethereumetl.token_prices_usd INNER JOIN ethereumetl.contracts ON contract_address=address
    WHERE asset_platform_id = 'ethereum';

-- COMMAND ----------

WITH CTE AS(
   SELECT address, name, price_usd, ROW_NUMBER() OVER(PARTITION BY address ORDER BY name) AS RN
   FROM tokens_silver
)
DELETE FROM CTE WHERE RN > 1

-- COMMAND ----------

SELECT * FROM tokens_silver;

-- COMMAND ----------

SELECT COUNT(*), COUNT(DISTINCT name), COUNT(DISTINCT address)
FROM tokens_silver;

-- COMMAND ----------

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
SELECT number, CAST(CAST(timestamp AS TIMESTAMP) AS DATE) AS date 
FROM blocks 
WHERE number = (SELECT MAX(number) FROM blocks);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: At what block did the first ERC20 transfer happen?

-- COMMAND ----------

USE ethereumetl;
SELECT MIN(block_number)
FROM token_transfers;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: How many ERC20 compatible contracts are there on the blockchain?

-- COMMAND ----------

USE ethereumetl;

SELECT COUNT(*)
FROM contracts C INNER JOIN tokens T ON C.address=T.address;

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Q: What percentage of transactions are calls to contracts

-- COMMAND ----------

-- TODO replace with CASE WHEN to_address is a contract address
USE ethereumetl;
SELECT AVG(CASE WHEN to_address='' THEN 1.0 ELSE 0.0 END)*100 AS contractCallPercentage
FROM transactions;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What are the top 100 tokens based on transfer count?

-- COMMAND ----------

USE ethereumetl;

SELECT COUNT(*) AS frequency, token_address, name
FROM (token_transfers LEFT JOIN tokens ON token_address=address)
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
  FROM token_transfers
  GROUP BY to_address
);

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
SELECT (SUM(value)/(1000000000000000000)) AS EtherVolume
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

-- TODO CHECK

USE ethereumetl;

SELECT MAX(count)
FROM (
      SELECT transaction_hash, COUNT(*) AS count
      FROM token_transfers
      GROUP BY transaction_hash
      );

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: Token balance for any address on any date?

-- COMMAND ----------

SELECT to_address FROM transactions LIMIT 100;

-- COMMAND ----------

SET tmp=${wallet_address};

-- COMMAND ----------

SELECT number
FROM blocks
WHERE CAST(CAST(timestamp AS TIMESTAMP) AS DATE) > CAST('${start_date}' AS DATE)
ORDER BY timestamp ASC;

-- COMMAND ----------

USE ethereumetl;

--= 0x5df9b87991262f6ba471f09758cde1c0fc1de734;
--SET startDate = str_to_date('2017/01/01', '%Y/%m/%d');

--SELECT from_address, SUM(value) AS sold
--FROM token_transfers
--WHERE from_address = addr AND block_number < (SELECT MIN(number)
--                                              FROM blocks
--                                              WHERE CAST(CAST(timestamp AS TIMESTAMP) AS DATE) > startDate
--                                              )
--GROUP BY from_address;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Viz the transaction count over time (network use)

-- COMMAND ----------

USE ethereumetl;

SELECT CAST(CAST(timestamp AS TIMESTAMP) AS DATE) AS date, transaction_count
FROM blocks
WHERE transaction_count > 0;

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
