# Databricks notebook source
from pyspark.sql.functions import rand, when, col
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC # Scenario 1
# MAGIC Randomly replacing 40% of its unit_ids with Null in the Trips dataset for a given date (2026-01-16)

# COMMAND ----------

df = spark.sql("select * from bms_ds_bronze.trips where date = '2026-02-06'")
display(df.limit(10))
print(df.count())

# COMMAND ----------

df = df.withColumn(
    "unit_id",
    when(rand() < 0.4, None).otherwise(col("unit_id"))
)

display(df.limit(10))
print(df.count())

# COMMAND ----------

(
    df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_test1")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Scenario 2
# MAGIC Mixed evaluation 35% of rows have null gps_coverage and start_lat/long values (date: 2026-02-09)

# COMMAND ----------

df = spark.sql("select * from bms_ds_bronze.trips where date = '2026-02-09'")
display(df.limit(10))
print(df.count())

# COMMAND ----------

distance_null_rate = 0.35
gps_coverage_threshold = 0.5
seed = 42

# Create a single failure flag, then apply to all columns
faulty_df = df.withColumn(
    "_gps_failure", 
    F.rand(seed) < distance_null_rate  # Same random draw for all columns
)

faulty_df = faulty_df.withColumn(
    "distance",
    F.when(F.col("_gps_failure"), F.lit(None)).otherwise(F.col("distance"))
).withColumn(
    "start_longitude",
    F.when(F.col("_gps_failure"), F.lit(None)).otherwise(F.col("start_longitude"))
).withColumn(
    "start_latitude",
    F.when(F.col("_gps_failure"), F.lit(None)).otherwise(F.col("start_latitude"))
).drop("_gps_failure")

# COMMAND ----------

display(faulty_df)

# COMMAND ----------

(
    faulty_df
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_test1")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Scenario 3
# MAGIC CASE: Fuel Sensor Drift
# MAGIC Simulates gradual sensor drift where fuel readings are 2-3x higher than actual. This will cause ML model prediction errors and high fuel efficiency metrics.

# COMMAND ----------

df = spark.sql("select * from bms_ds_bronze.trips where date = '2026-02-10'")
display(df.limit(10))
print(df.count())

# COMMAND ----------

# DBTITLE 1,Cell 13
fuel_drift_rate = 0.15
drift_multiplier_low = 2.0
drift_multiplier_high = 3.5
seed = 42

# Create drift flag for correlated effect
faulty_df = df.withColumn(
    "_fuel_drift",
    F.rand(seed) < fuel_drift_rate
)

# Apply drift: multiply fuel by random factor between 2x-3.5x
faulty_df = faulty_df.withColumn(
    "fuel_consumption",
    F.when(
        F.col("_fuel_drift"),
        (F.col("fuel_consumption") * (drift_multiplier_low + F.rand(seed + 1) * (drift_multiplier_high - drift_multiplier_low))).cast("long")
    ).otherwise(F.col("fuel_consumption"))
).drop("_fuel_drift")

# Expected effects:
# - raw_fuel_consumption_mean: Increases ~15% * 2.75x = ~40% overall increase
# - silver_ml_large_error_count: Spikes (ML predictions vs actual differ)
# - p95_fuel_per_100km: Increases significantly

# COMMAND ----------

(
    faulty_df
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_test1")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Scenario 4
# MAGIC CASE: Timestamp Clock Skew
# MAGIC Simulates device clock synchronization failure. 10% of trips have timestamps 7+ days in future, 10% have timestamps 30+ days in past. This causes duration calculation errors and validation drops.

# COMMAND ----------

df = spark.sql("select * from bms_ds_bronze.trips where date = '2026-02-11'")
print(df.count())

# COMMAND ----------

# DBTITLE 1,Cell 17
future_skew_rate = 0.10
past_skew_rate = 0.10
future_days = 7
past_days = 30
seed = 42

# Create separate flags for future and past skew
faulty_df = df.withColumn(
    "_clock_skew_type",
    F.when(F.rand(seed) < future_skew_rate, F.lit("future"))
     .when(F.rand(seed + 1) < past_skew_rate, F.lit("past"))
     .otherwise(F.lit("normal"))
)

# Apply timestamp skew to end
faulty_df = faulty_df.withColumn(
    "end",
    F.when(
        F.col("_clock_skew_type") == "future",
        F.date_add(F.col("end"), future_days)
    ).when(
        F.col("_clock_skew_type") == "past",
        F.date_sub(F.col("end"), past_days)
    ).otherwise(F.col("end"))
)

# Also skew start for past cases (creates negative durations)
faulty_df = faulty_df.withColumn(
    "start",
    F.when(
        F.col("_clock_skew_type") == "past",
        F.date_add(F.col("start"), past_days + 1)  # Start AFTER end = negative duration
    ).otherwise(F.col("start"))
).drop("_clock_skew_type")

# Expected effects:
# - raw_max_trip_end_ts: Shows future dates
# - bronze_rows_dropped_by_rules: Spikes (invalid durations dropped)
# - bronze_survival_rate: Drops significantly
# - bronze_duration_mean: Becomes erratic

# COMMAND ----------

display(faulty_df.limit(30))

# COMMAND ----------

(
    faulty_df
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_test1")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Scenario 5
# MAGIC CASE: Timestamp Clock Skew
# MAGIC Simulates device clock synchronization failure. 10% of trips have timestamps 7+ days in future, 10% have timestamps 30+ days in past. This causes duration calculation errors and validation drops.