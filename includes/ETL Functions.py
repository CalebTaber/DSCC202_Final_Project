# Databricks notebook source
def create_tokens_silver():
    from pyspark.sql.types import _parse_datatype_string

    tpu = spark.table("ethereumetl.token_prices_usd")
    tokens = spark.table("ethereumetl.tokens")
    
    assert tpu.schema == _parse_datatype_string("id: string, symbol: string, name: string, asset_platform_id: string, description: string, links: string, image: string, contract_address: string, sentiment_votes_up_percentage: double, sentiment_votes_down_percentage: double, market_cap_rank: double, coingecko_rank: double, coingecko_score: double, developer_score: double, community_score: double, liquidity_score: double, public_interest_score: double, price_usd: double"), "Schema is not validated"
    print("TPU assertion passed")
    
    assert tokens.schema == _parse_datatype_string("address: string, symbol: string, name: string, decimals: bigint, total_supply: decimal(38,0), start_block: bigint, end_block: bigint"), "Schema is not validated"
    print("Tokens assertion passed")
    
    
    tpu = spark.table("ethereumetl.token_prices_usd")
    tokens = spark.table("ethereumetl.tokens")
    
    tokens_silver = (
                     (tpu.join(tokens, (tpu.contract_address == tokens.address), 'inner')
                         .where(tpu.asset_platform_id == 'ethereum')) 
                     .select(col('contract_address').alias('address'),
                                  col('ethereumetl.tokens.name'),
                                  col('ethereumetl.tokens.symbol'),
                                  col('price_usd'))
                     .dropDuplicates(['address']) 
                     .withColumn("id", row_number().over(Window.orderBy(lit(1))))
                    )
    
    return tokens_silver
    
    

    
def create_tt_silver ():
    tok_trans_sub = spark.table('ethereumetl.token_transfers').select('token_address',  'from_address', 'to_address', 'value', 'block_number')
    blocks_sub = spark.table('ethereumetl.blocks').select('timestamp', 'number')
    tokens_silver_sub = spark.table('g04_db.toks_silver').select('address', 'id')
    
    assert tok_trans_sub.schema == _parse_datatype_string("token_address:string, from_address:string,to_address:string,value:decimal(38,0),block_number:long"), "tok_trans_sub schema is not validated"
    print("tok_trans_sub assertion passed")
    
    assert blocks_sub.schema == _parse_datatype_string("timestamp:long, number:long"), "blocks_sub schema is not validated"
    print("blocks_sub assertion passed")
    
    assert tokens_silver_sub.schema == _parse_datatype_string("address:string, id:integer"), "tokens_silver_sub schema is not validated"
    print("tokens_silver_sub assertion passed")
    
    sqlContext.setConf("spark.sql.shuffle.partitions","auto")

    tok_trans_sub = spark.table('ethereumetl.token_transfers').select('token_address', 'from_address', 'to_address', 'value', 'block_number')
    blocks_sub = spark.table('ethereumetl.blocks').select('timestamp', 'number')
    tokens_silver_sub = spark.table('g04_db.toks_silver').select('address', 'id')
    
    tt_silver = (
                 tok_trans_sub.join(tokens_silver_sub, (tokens_silver_sub.address == tok_trans_sub.token_address), 'inner')
                              .select('id', 'to_address', 'from_address', 'value', 'block_number')
                              .where(tokens_silver_sub.address == tok_trans_sub.token_address)
                              .join(blocks_sub, (tok_trans_sub.block_number == blocks_sub.number), 'inner')
                              .where(tok_trans_sub.block_number == blocks_sub.number)
                              .select('id', 'to_address', 'from_address', 'value', 'timestamp')
                              .withColumn("timestamp", col('timestamp').cast("timestamp"))
                  )
    
    return tt_silver


def create_triple():
    from pyspark.sql.window import Window
    from pyspark.sql.functions import sum
    
    tt_silver = spark.table('g04_db.tt_silver_part')
    toks_silver = spark.table('g04_db.toks_silver')
    
    bought = tt_silver.withColumnRenamed('to_address', 'to_addr')\
                               .groupBy('id', 'to_addr')\
                               .agg(sum('value').alias('b_val'))
    
    sold = tt_silver.withColumnRenamed('from_address', 'from_addr')\
                             .groupBy('id', 'from_addr')\
                             .agg(sum('value').alias('s_val'))
    
    with_bal = bought.join(sold, (bought.id == sold.id) & (bought.to_addr == sold.from_addr), 'inner')\
                     .withColumn('balance', bought.b_val - sold.s_val)\
                     .withColumnRenamed('to_addr', 'addr')\
                     .select('addr', bought.id, 'balance')
    
    with_usd = with_bal.join(toks_silver, with_bal.id == toks_silver.id, 'inner')\
                       .withColumn('bal_usd', with_bal.balance * toks_silver.price_usd)\
                       .select(with_bal.addr, with_bal.id, 'bal_usd')
    
    addrs_abridged = with_usd.select('addr').distinct()\
                             .withColumn("addr_id", row_number().over(Window.orderBy(lit(1))))
    
    triple = with_usd.join(addrs_abridged, with_usd.addr == addrs_abridged.addr, 'inner')\
                     .select(addrs_abridged.addr_id, with_usd.id, 'bal_usd')\
                     .withColumnRenamed('id', 'tok_id')

    return triple
