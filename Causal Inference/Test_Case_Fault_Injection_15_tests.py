# Databricks notebook source
# MAGIC %md
# MAGIC # Test Case Fault Injection for RCA Evaluation
# MAGIC
# MAGIC This notebook injects 11 additional fault test cases (Cases 5-15) into temp tables.
# MAGIC The 4 baseline cases (1-4) are already created.
# MAGIC
# MAGIC **Total Test Cases: 15 (3 per category)**
# MAGIC
# MAGIC Categories:
# MAGIC 1. Raw Data Quality - NULL Injections (Cases 1, 5, 6)
# MAGIC 2. Raw Data Quality - Validation Issues (Cases 2, 7, 8)
# MAGIC 3. ML Layer - Model Quality (Cases 3, 9, 10)
# MAGIC 4. Temporal/Duration Issues (Cases 4, 11, 12)
# MAGIC 5. Bronze Layer - Transformation Issues (Cases 13, 14, 15)

# COMMAND ----------

from pyspark.sql.functions import rand, when, col, lit, expr, concat
import pyspark.sql.functions as F
from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Test case configuration with dates and parameters
TEST_CASES = {
    # CATEGORY 1: Raw Data Quality - NULL Injections
    "case1": {"date": "2026-02-06", "name": "unit_id nulls", "null_rate": 0.40},
    "case5": {"date": "2025-10-01", "name": "sensor_reading nulls", "null_rate": 0.35},
    "case6": {"date": "2025-10-02", "name": "vehicle_id nulls", "null_rate": 0.25},
    
    # CATEGORY 2: Raw Data Quality - Validation Issues
    "case2": {"date": "2026-02-09", "name": "distance/gps nulls", "null_rate": 0.35},
    "case7": {"date": "2025-10-03", "name": "extreme values", "severity": 0.20},
    "case8": {"date": "2025-10-04", "name": "invalid ranges", "null_rate": 0.30},
    
    # CATEGORY 3: ML Layer - Model Quality
    "case3": {"date": "2026-02-10", "name": "fuel sensor drift", "drift_rate": 0.15},
    "case9": {"date": "2025-10-05", "name": "speed prediction drift", "drift_rate": 0.20},
    "case10": {"date": "2025-10-06", "name": "prediction outliers", "outlier_rate": 0.15},
    
    # CATEGORY 4: Temporal/Duration Issues
    "case4": {"date": "2026-02-11", "name": "clock skew", "skew_rate": 0.10},
    "case11": {"date": "2025-10-07", "name": "duration anomalies", "anomaly_rate": 0.20},
    "case12": {"date": "2025-10-08", "name": "timestamp inconsistencies", "inconsistency_rate": 0.25},
    
    # CATEGORY 5: Bronze Layer - Transformation Issues
    "case13": {"date": "2025-10-09", "name": "aggregation errors", "error_rate": 0.15},
    "case14": {"date": "2025-10-10", "name": "join failures", "failure_rate": 0.20},
    "case15": {"date": "2025-10-11", "name": "duplicate handling", "dup_rate": 0.30},

    # Less severe dq issues 
    "case16": {"date": "2025-10-12", "name": "vehicle_id nulls", "null_rate": 0.05},
    "case17": {"date": "2025-10-13", "name": "extreme values", "severity": 0.05},
    "case18": {"date": "2025-10-14", "name": "invalid ranges", "null_rate": 0.10},
    "case19": {"date": "2025-10-15", "name": "speed prediction drift", "drift_rate": 0.10},
    "case20": {"date": "2025-10-16", "name": "prediction outliers", "outlier_rate": 0.05},
    "case21": {"date": "2025-10-17", "name": "duration anomalies", "anomaly_rate": 0.10},
    "case22": {"date": "2025-10-18", "name": "timestamp inconsistencies", "inconsistency_rate": 0.15},
}

print("="*70)
print("TEST CASE CONFIGURATION")
print("="*70)
print(f"\nTotal cases: {len(TEST_CASES)}")
for case_id, config in TEST_CASES.items():
    print(f"  {case_id.upper()}: {config['name']} ({config['date']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## CATEGORY 1: Raw Data Quality - NULL Injections
# MAGIC Cases: 1 (existing), 5, 6

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 5: Sensor Reading Nulls for all Fuel Consumption columns
# MAGIC **Domain:** Raw sensor data quality
# MAGIC **Fault:** Sensor calibration failure causing null readings
# MAGIC **Expected Metrics Affected:**
# MAGIC - fuel_consumption_ecol/ecor/fms_high becomes null

# COMMAND ----------

# DBTITLE 1,Cell 7
case5_config = TEST_CASES["case5"]
df_case5 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case5_config['date']}'")

print(f"Case 5: {case5_config['name']} ({case5_config['date']})")
print(f"Original count: {df_case5.count()}")

# Inject nulls into fuel_consumption column (35% rate)
df_case5 = df_case5.withColumn("fuel_consumption_ecol", lit(None))
df_case5 = df_case5.withColumn("fuel_consumption_ecor", lit(None))
df_case5 = df_case5.withColumn("fuel_consumption_fms_high", lit(None))


print(f"Rows with null fuel_consumption: {df_case5.filter(col('fuel_consumption_ecol').isNull()).count()}")
print(f"Rows with null fuel_consumption: {df_case5.filter(col('fuel_consumption_ecor').isNull()).count()}")
print(f"Rows with null fuel_consumption: {df_case5.filter(col('fuel_consumption_fms_high').isNull()).count()}")



(
    df_case5
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 5 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 6: Vehicle ID Nulls (25% rate)
# MAGIC **Domain:** Raw data foreign key quality
# MAGIC **Fault:** Vehicle master data sync failure
# MAGIC **Expected Metrics Affected:**
# MAGIC - raw_null_count_vehicle_id increases
# MAGIC - silver_vehicle_info_join_miss_rate increases (during bronze→silver join)
# MAGIC - silver_vehicle_type_nulls increases

# COMMAND ----------

case6_config = TEST_CASES["case6"]
df_case6 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case6_config['date']}'")

print(f"Case 6: {case6_config['name']} ({case6_config['date']})")
print(f"Original count: {df_case6.count()}")

# Inject nulls into vehicle_id column (25% rate)
df_case6 = df_case6.withColumn(
    "unit_id",
    when(rand() < case6_config['null_rate'], None).otherwise(col("unit_id"))
)

print(f"Rows with null vehicle_id: {df_case6.filter(col('unit_id').isNull()).count()}")

(
    df_case6
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 6 saved")

# COMMAND ----------

case16_config = TEST_CASES["case16"]
df_case16 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case16_config['date']}'")

print(f"Case 16: {case16_config['name']} ({case16_config['date']})")
print(f"Original count: {df_case16.count()}")
print('severity:', case16_config['null_rate'])

# Inject nulls into vehicle_id column (25% rate)
df_case16 = df_case16.withColumn(
    "unit_id",
    when(rand() < case16_config['null_rate'], None).otherwise(col("unit_id"))
)

print(f"Rows with null vehicle_id: {df_case16.filter(col('unit_id').isNull()).count()}")

(
    df_case16
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 16 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## CATEGORY 2: Raw Data Quality - Validation Issues
# MAGIC Cases: 2 (existing), 7, 8

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 7: Extreme Values (20% of rows have extreme values)
# MAGIC **Domain:** Raw data outliers/anomalies
# MAGIC **Fault:** Sensor reading noise or calibration drift
# MAGIC **Expected Metrics Affected:**
# MAGIC - raw_avg_speed_mean increases significantly
# MAGIC - raw_fuel_consumption_std increases
# MAGIC - bronze_invalid_avg_speed_rows increases
# MAGIC - bronze_rows_dropped_by_rules increases

# COMMAND ----------

# DBTITLE 1,Cell 12
case7_config = TEST_CASES["case7"]
df_case7 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case7_config['date']}'")

print(f"Case 7: {case7_config['name']} ({case7_config['date']})")
print(f"Original count: {df_case7.count()}")

seed = 42
extreme_multiplier = 10.0  # Make values 10x normal

# Inject extreme values into speed and fuel columns
df_case7 = df_case7.withColumn(
    "_inject_extreme",
    rand(seed) < case7_config['severity']
).withColumn(
    "avg_speed",
    when(col("_inject_extreme"), (col("avg_speed") * extreme_multiplier).cast("long")).otherwise(col("avg_speed"))
).withColumn(
    "fuel_consumption",
    when(col("_inject_extreme"), (col("fuel_consumption_ecol") * extreme_multiplier).cast("long")).otherwise(col("fuel_consumption_ecol"))
).withColumn(
    "fuel_consumption_ecor",
    when(col("_inject_extreme"), (col("fuel_consumption_ecol") * extreme_multiplier).cast("long")).otherwise(col("fuel_consumption_ecol"))
).withColumn(
    "fuel_consumption_fms_high",
    when(col("_inject_extreme"), (col("fuel_consumption_ecol") * extreme_multiplier).cast("long")).otherwise(col("fuel_consumption_ecol"))
).drop("_inject_extreme")

print(f"Rows with extreme values: {int(df_case7.count() * case7_config['severity'])}")

(
    df_case7
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 7 saved")

# COMMAND ----------

case17_config = TEST_CASES["case17"]
df_case17 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case17_config['date']}'")

print(f"Case 7: {case17_config['name']} ({case17_config['date']})")
print(f"Original count: {df_case17.count()}")

seed = 42
extreme_multiplier = 10.0  # Make values 10x normal

# Inject extreme values into speed and fuel columns
df_case17 = df_case17.withColumn(
    "_inject_extreme",
    rand(seed) < case17_config['severity']
).withColumn(
    "avg_speed",
    when(col("_inject_extreme"), (col("avg_speed") * extreme_multiplier).cast("long")).otherwise(col("avg_speed"))
).withColumn(
    "fuel_consumption",
    when(col("_inject_extreme"), (col("fuel_consumption_ecol") * extreme_multiplier).cast("long")).otherwise(col("fuel_consumption_ecol"))
).withColumn(
    "fuel_consumption_ecor",
    when(col("_inject_extreme"), (col("fuel_consumption_ecol") * extreme_multiplier).cast("long")).otherwise(col("fuel_consumption_ecol"))
).withColumn(
    "fuel_consumption_fms_high",
    when(col("_inject_extreme"), (col("fuel_consumption_ecol") * extreme_multiplier).cast("long")).otherwise(col("fuel_consumption_ecol"))
).drop("_inject_extreme")

print(f"Rows with extreme values: {int(df_case17.count() * case17_config['severity'])}")

(
    df_case17
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 17 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 8: Invalid Ranges (30% of rows)
# MAGIC **Domain:** Raw data constraint violations
# MAGIC **Fault:** Data entry errors or parsing failures
# MAGIC **Expected Metrics Affected:**
# MAGIC - raw_duration_mean becomes negative/erratic
# MAGIC - bronze_rows_dropped_by_rules increases
# MAGIC - bronze_survival_rate decreases
# MAGIC - raw_negative_duration_count increases

# COMMAND ----------

case8_config = TEST_CASES["case8"]
df_case8 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case8_config['date']}'")

print(f"Case 8: {case8_config['name']} ({case8_config['date']})")
print(f"Original count: {df_case8.count()}")

seed = 43

# Create invalid ranges: negative durations, future end dates, etc.
df_case8 = df_case8.withColumn(
    "_inject_invalid",
    rand(seed) < case8_config['null_rate']
).withColumn(
    "start",
    when(col("_inject_invalid"), expr("date_add(end, 1)")).otherwise(col("start"))  # Start after end
).drop("_inject_invalid")

print(f"Rows with invalid ranges: {int(df_case8.count() * case8_config['null_rate'])}")

(
    df_case8
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 8 saved")

# COMMAND ----------

case18_config = TEST_CASES["case18"]
df_case18 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case18_config['date']}'")

print(f"Case 8: {case18_config['name']} ({case18_config['date']})")
print(f"Original count: {df_case18.count()}")

seed = 43
print(case18_config['null_rate'])
# Create invalid ranges: negative durations, future end dates, etc.
df_case18 = df_case18.withColumn(
    "_inject_invalid",
    rand(seed) < case18_config['null_rate']
).withColumn(
    "start",
    when(col("_inject_invalid"), expr("date_add(end, 1)")).otherwise(col("start"))  # Start after end
).drop("_inject_invalid")

print(f"Rows with invalid ranges: {int(df_case18.count() * case18_config['null_rate'])}")

(
    df_case18
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 18 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## CATEGORY 3: ML Layer - Model Quality
# MAGIC Cases: 3 (existing), 9, 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 9: Speed Prediction Drift (20% of trips)
# MAGIC **Domain:** ML model degradation
# MAGIC **Fault:** Speed model overfits to training distribution, fails on test
# MAGIC **Expected Metrics Affected:**
# MAGIC - silver_ml_large_error_count increases
# MAGIC - silver_ml_prediction_std increases
# MAGIC - silver_ml_residual_mean becomes non-zero
# MAGIC - p95_speed increases

# COMMAND ----------

# DBTITLE 1,Cell 17
case9_config = TEST_CASES["case9"]
df_case9 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case9_config['date']}'")

print(f"Case 9: {case9_config['name']} ({case9_config['date']})")
print(f"Original count: {df_case9.count()}")

seed = 44
drift_multiplier_low = 1.5
drift_multiplier_high = 2.5

# Inject speed drift (simulates model prediction error)
df_case9 = df_case9.withColumn(
    "_speed_drift",
    rand(seed) < case9_config['drift_rate']
).withColumn(
    "avg_speed",
    when(
        col("_speed_drift"),
        (col("avg_speed") * (drift_multiplier_low + rand(seed + 1) * (drift_multiplier_high - drift_multiplier_low))).cast("long")
    ).otherwise(col("avg_speed"))
).drop("_speed_drift")

print(f"Rows with speed drift: {int(df_case9.count() * case9_config['drift_rate'])}")

(
    df_case9
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 9 saved")

# COMMAND ----------

case19_config = TEST_CASES["case19"]
df_case19 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case19_config['date']}'")

print(f"Case 9: {case19_config['name']} ({case19_config['date']})")
print(f"Original count: {df_case19.count()}")

seed = 44
drift_multiplier_low = 1.5
drift_multiplier_high = 2.5

# Inject speed drift (simulates model prediction error)
df_case19 = df_case19.withColumn(
    "_speed_drift",
    rand(seed) < case19_config['drift_rate']
).withColumn(
    "avg_speed",
    when(
        col("_speed_drift"),
        (col("avg_speed") * (drift_multiplier_low + rand(seed + 1) * (drift_multiplier_high - drift_multiplier_low))).cast("long")
    ).otherwise(col("avg_speed"))
).drop("_speed_drift")

print(f"Rows with speed drift: {int(df_case19.count() * case19_config['drift_rate'])}")

(
    df_case19
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 19 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 10: Prediction Outliers (15% of rows have extreme predictions)
# MAGIC **Domain:** ML model outliers
# MAGIC **Fault:** Rare edge cases not covered in training
# MAGIC **Expected Metrics Affected:**
# MAGIC - silver_ml_large_error_count increases significantly
# MAGIC - silver_ml_imputed_* values spike
# MAGIC - bronze_invalid_avg_speed_rows increases (if prediction validation fails)

# COMMAND ----------

# DBTITLE 1,Cell 19
case10_config = TEST_CASES["case10"]
df_case10 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case10_config['date']}'")

print(f"Case 10: {case10_config['name']} ({case10_config['date']})")
print(f"Original count: {df_case10.count()}")

seed = 45
outlier_multiplier = 20.0  # Extreme predictions

# Inject prediction outliers
df_case10 = df_case10.withColumn(
    "_has_outlier",
    rand(seed) < case10_config['outlier_rate']
).withColumn(
    "fuel_consumption_ecol",
    when(col("_has_outlier"), (col("fuel_consumption") * outlier_multiplier).cast("long")).otherwise(col("fuel_consumption"))
).withColumn(
    "fuel_consumption_ecor",
    when(col("_has_outlier"), (col("fuel_consumption") * outlier_multiplier).cast("long")).otherwise(col("fuel_consumption"))
).withColumn(
    "fuel_consumption_fms_high",
    when(col("_has_outlier"), (col("fuel_consumption") * outlier_multiplier).cast("long")).otherwise(col("fuel_consumption"))
).drop("_has_outlier")

print(f"Rows with prediction outliers: {int(df_case10.count() * case10_config['outlier_rate'])}")

(
    df_case10
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 10 saved")

# COMMAND ----------

case20_config = TEST_CASES["case20"]
df_case20 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case20_config['date']}'")

print(f"Case 10: {case20_config['name']} ({case20_config['date']})")
print(f"Original count: {df_case20.count()}")

seed = 45
outlier_multiplier = 20.0  # Extreme predictions

# Inject prediction outliers
df_case20 = df_case20.withColumn(
    "_has_outlier",
    rand(seed) < case20_config['outlier_rate']
).withColumn(
    "fuel_consumption_ecol",
    when(col("_has_outlier"), (col("fuel_consumption") * outlier_multiplier).cast("long")).otherwise(col("fuel_consumption"))
).withColumn(
    "fuel_consumption_ecor",
    when(col("_has_outlier"), (col("fuel_consumption") * outlier_multiplier).cast("long")).otherwise(col("fuel_consumption"))
).withColumn(
    "fuel_consumption_fms_high",
    when(col("_has_outlier"), (col("fuel_consumption") * outlier_multiplier).cast("long")).otherwise(col("fuel_consumption"))
).drop("_has_outlier")

print(f"Rows with prediction outliers: {int(df_case20.count() * case20_config['outlier_rate'])}")

(
    df_case20
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 20 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## CATEGORY 4: Temporal/Duration Issues
# MAGIC Cases: 4 (existing), 11, 12

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 11: Duration Anomalies (20% of rows have anomalous durations)
# MAGIC **Domain:** Temporal data quality
# MAGIC **Fault:** GPS tracking loss during trip causes incorrect duration recording
# MAGIC **Expected Metrics Affected:**
# MAGIC - raw_duration_mean becomes erratic
# MAGIC - raw_duration_std increases
# MAGIC - bronze_rows_dropped_by_rules increases (validation fails)
# MAGIC - bronze_survival_rate decreases

# COMMAND ----------

case11_config = TEST_CASES["case11"]
df_case11 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case11_config['date']}'")

print(f"Case 11: {case11_config['name']} ({case11_config['date']})")
print(f"Original count: {df_case11.count()}")

seed = 46
anomaly_hours = 6  # Add random hours to duration

# Inject duration anomalies
df_case11 = df_case11.withColumn(
    "_has_duration_anomaly",
    rand(seed) < case11_config['anomaly_rate']
).withColumn(
    "end",
    when(
        col("_has_duration_anomaly"),
        expr(f"date_add(end, {anomaly_hours})")
    ).otherwise(col("end"))
).drop("_has_duration_anomaly")

print(f"Rows with duration anomalies: {int(df_case11.count() * case11_config['anomaly_rate'])}")

(
    df_case11
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 11 saved")

# COMMAND ----------

case21_config = TEST_CASES["case21"]
df_case21 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case21_config['date']}'")

print(f"Case 11: {case21_config['name']} ({case21_config['date']})")
print(f"Original count: {df_case21.count()}")

seed = 46
anomaly_hours = 6  # Add random hours to duration

# Inject duration anomalies
df_case21 = df_case21.withColumn(
    "_has_duration_anomaly",
    rand(seed) < case21_config['anomaly_rate']
).withColumn(
    "end",
    when(
        col("_has_duration_anomaly"),
        expr(f"date_add(end, {anomaly_hours})")
    ).otherwise(col("end"))
).drop("_has_duration_anomaly")

print(f"Rows with duration anomalies: {int(df_case21.count() * case21_config['anomaly_rate'])}")

(
    df_case21
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 21 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 12: Timestamp Inconsistencies (25% of rows)
# MAGIC **Domain:** Temporal data consistency
# MAGIC **Fault:** Multiple device clocks out of sync
# MAGIC **Expected Metrics Affected:**
# MAGIC - raw_max_trip_start_ts, raw_max_trip_end_ts show inconsistencies
# MAGIC - bronze_duplicate_rows_removed increases
# MAGIC - silver_vehicle_info_join_miss_rate increases (time-based join failures)

# COMMAND ----------

case12_config = TEST_CASES["case12"]
df_case12 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case12_config['date']}'")

print(f"Case 12: {case12_config['name']} ({case12_config['date']})")
print(f"Original count: {df_case12.count()}")

seed = 47

# Inject timestamp inconsistencies (both start and end offset)
df_case12 = df_case12.withColumn(
    "_has_inconsistency",
    rand(seed) < case12_config['inconsistency_rate']
).withColumn(
    "start",
    when(
        col("_has_inconsistency"),
        expr("date_sub(start, 1)")  # Start 1 day earlier
    ).otherwise(col("start"))
).withColumn(
    "end",
    when(
        col("_has_inconsistency"),
        expr("date_add(end, 1)")  # End 1 day later
    ).otherwise(col("end"))
).drop("_has_inconsistency")

print(f"Rows with timestamp inconsistencies: {int(df_case12.count() * case12_config['inconsistency_rate'])}")

(
    df_case12
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 12 saved")

# COMMAND ----------

case22_config = TEST_CASES["case22"]
df_case22 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case22_config['date']}'")

print(f"Case 12: {case22_config['name']} ({case22_config['date']})")
print(f"Original count: {df_case22.count()}")

seed = 47

# Inject timestamp inconsistencies (both start and end offset)
df_case22 = df_case22.withColumn(
    "_has_inconsistency",
    rand(seed) < case22_config['inconsistency_rate']
).withColumn(
    "start",
    when(
        col("_has_inconsistency"),
        expr("date_sub(start, 1)")  # Start 1 day earlier
    ).otherwise(col("start"))
).withColumn(
    "end",
    when(
        col("_has_inconsistency"),
        expr("date_add(end, 1)")  # End 1 day later
    ).otherwise(col("end"))
).drop("_has_inconsistency")

print(f"Rows with timestamp inconsistencies: {int(df_case22.count() * case22_config['inconsistency_rate'])}")

(
    df_case22
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 22 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## CATEGORY 5: Bronze Layer - Transformation Issues
# MAGIC Cases: 13, 14, 15

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 13: Aggregation Errors (15% of rows)
# MAGIC **Domain:** Bronze transformation quality
# MAGIC **Fault:** Aggregation window misconfiguration causes duplicate counts
# MAGIC **Expected Metrics Affected:**
# MAGIC - bronze_duplicate_rows_removed increases
# MAGIC - bronze_rows_dropped_by_rules increases
# MAGIC - raw_unique_units may decrease (if deduped wrong)

# COMMAND ----------

# DBTITLE 1,Cell 27
case13_config = TEST_CASES["case13"]
df_case13 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case13_config['date']}'")

print(f"Case 13: {case13_config['name']} ({case13_config['date']})") 
print(f"Original count: {df_case13.count()}")

seed = 48

# Simulate aggregation errors by duplicating rows
df_case13_marked = df_case13.withColumn(
    "_has_aggregation_error",
    rand(seed) < case13_config['error_rate']
)

df_case13 = df_case13_marked.filter(
    ~col("_has_aggregation_error")  # Keep the base rows
).union(
    df_case13_marked.filter(col("_has_aggregation_error")).union(
        df_case13_marked.filter(col("_has_aggregation_error"))  # Duplicate these rows
    )
).drop("_has_aggregation_error")

print(f"Rows (after duplication): {df_case13.count()}")

(
    df_case13
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 13 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 14: Join Failures (20% join miss rate)
# MAGIC **Domain:** Bronze-to-Silver transformation
# MAGIC **Fault:** Vehicle master data incomplete, join keys don't match
# MAGIC **Expected Metrics Affected:**
# MAGIC - silver_vehicle_info_join_miss_rate increases significantly
# MAGIC - silver_vehicle_type_nulls increases
# MAGIC - bronze_null_primary_key_rows may increase

# COMMAND ----------

case14_config = TEST_CASES["case14"]
df_case14 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case14_config['date']}'")

print(f"Case 14: {case14_config['name']} ({case14_config['date']})")
print(f"Original count: {df_case14.count()}")

seed = 49

# Simulate join failures by corrupting vehicle_id (so join won't match)
df_case14 = df_case14.withColumn(
    "_has_join_failure",
    rand(seed) < case14_config['failure_rate']
).withColumn(
    "unit_id",
    when(
        col("_has_join_failure"),
        concat(col("unit_id"), lit("qwertyuiop"))  # Corrupt ID so join fails
    ).otherwise(col("unit_id"))
).drop("_has_join_failure")

print(f"Rows with join failures: {int(df_case14.count() * case14_config['failure_rate'])}")

(
    df_case14
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 14 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CASE 15: Duplicate Handling (30% of rows duplicated)
# MAGIC **Domain:** Bronze data deduplication
# MAGIC **Fault:** Duplicate record processing fails, both versions reach KPIs
# MAGIC **Expected Metrics Affected:**
# MAGIC - bronze_duplicate_rows_removed increases
# MAGIC - raw_unique_units decreases (same trip counted multiple times)
# MAGIC - silver_avg_speed may become erratic (conflicting duplicates)

# COMMAND ----------

case15_config = TEST_CASES["case15"]
df_case15 = spark.sql(f"select * from bms_ds_bronze.trips where date = '{case15_config['date']}'")

print(f"Case 15: {case15_config['name']} ({case15_config['date']})")
print(f"Original count: {df_case15.count()}")

seed = 50

# Create duplicates
df_case15_dup = df_case15.filter(rand(seed) < case15_config['dup_rate'])
df_case15 = df_case15.union(df_case15_dup)  # Add duplicates back

print(f"Rows (after duplication): {df_case15.count()}")

(
    df_case15
    .write
    .mode("append")
    .format("delta")
    .saveAsTable("bms_ds_prod.bms_ds_dasc.temp_fault_injection_training")
)

print("✓ Case 15 saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("="*70)
print("TEST CASE INJECTION COMPLETE")
print("="*70)
print(f"\n✓ All 15 test cases injected into temp_fault_injection_training table")
print(f"\nNext steps:")
print(f"1. Run ETL pipeline on these dates to generate metrics")
print(f"2. Run RCA evaluation notebook with these test cases")
print(f"3. Evaluate causal discovery accuracy")

final_count = spark.sql("select count(distinct date) as date_count, count(*) as row_count from bms_ds_prod.bms_ds_dasc.temp_fault_injection_training").collect()
print(f"\nTemp table stats:")
for row in final_count:
    print(f"  Unique dates: {row['date_count']}")
    print(f"  Total rows: {row['row_count']}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix

# COMMAND ----------

# MAGIC %sql
# MAGIC select metric_name
# MAGIC from bms_ds_dasc.temp_raw_metrics
# MAGIC group by metric_name
# MAGIC having min(metric_value) = 0 and max(metric_value) = 0

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from bms_ds_dasc.temp_raw_metrics
# MAGIC where date = '2026-02-06' 
# MAGIC and pipeline_stage = 'raw'

# COMMAND ----------

# %sql
# insert into bms_ds_dasc.temp_raw_metrics
# select * from (select 
#   date,
#   "raw" as pipeline_stage,
#   metric_name,
#   metric_value, 
#   now() as created_at 
# from (
#   select 
#     date,
#     mean(fuel_consumption_ecol) as mean_fuel_consumption_ecol,
#     std(fuel_consumption_ecol) as std_fuel_consumption_ecol,
#     mean(fuel_consumption_ecor) as mean_fuel_consumption_ecor,
#     std(fuel_consumption_ecor) as std_fuel_consumption_ecor,
#     mean(fuel_consumption_fms_high) as mean_fuel_consumption_fms_high,
#     std(fuel_consumption_fms_high) as std_fuel_consumption_fms_high
#   from bms_ds_prod.bms_ds_dasc.temp_telematics_raw
#   group by date
# ) t
# unpivot (
#   metric_value for metric_name in (
#     mean_fuel_consumption_ecol,
#     std_fuel_consumption_ecol,
#     mean_fuel_consumption_ecor,
#     std_fuel_consumption_ecor,
#     mean_fuel_consumption_fms_high,
#     std_fuel_consumption_fms_high
#   )
# ))

# union all

# (select 
#   date,
#   "bronze" as pipeline_stage,
#   metric_name,
#   metric_value, 
#   now() as created_at 
# from (
#   select 
#     date,
#     mean(fuel_consumption_ecol) as mean_fuel_consumption_ecol,
#     std(fuel_consumption_ecol) as std_fuel_consumption_ecol,
#     mean(fuel_consumption_ecor) as mean_fuel_consumption_ecor,
#     std(fuel_consumption_ecor) as std_fuel_consumption_ecor,
#     mean(fuel_consumption_fms_high) as mean_fuel_consumption_fms_high,
#     std(fuel_consumption_fms_high) as std_fuel_consumption_fms_high
#   from bms_ds_prod.bms_ds_dasc.temp_telematics_bronze
#   group by date
# ) t
# unpivot (
#   metric_value for metric_name in (
#     mean_fuel_consumption_ecol,
#     std_fuel_consumption_ecol,
#     mean_fuel_consumption_ecor,
#     std_fuel_consumption_ecor,
#     mean_fuel_consumption_fms_high,
#     std_fuel_consumption_fms_high
#   )
# ))

# COMMAND ----------

# DBTITLE 1,Untitled
# MAGIC %sql
# MAGIC select date, pipeline_stage, count(*) from ((select 
# MAGIC   date,
# MAGIC   "raw" as pipeline_stage,
# MAGIC   metric_name,
# MAGIC   metric_value, 
# MAGIC   now() as created_at 
# MAGIC from (
# MAGIC   select 
# MAGIC     date,
# MAGIC     mean(fuel_consumption_ecol) as mean_fuel_consumption_ecol,
# MAGIC     std(fuel_consumption_ecol) as std_fuel_consumption_ecol,
# MAGIC     mean(fuel_consumption_ecor) as mean_fuel_consumption_ecor,
# MAGIC     std(fuel_consumption_ecor) as std_fuel_consumption_ecor,
# MAGIC     mean(fuel_consumption_fms_high) as mean_fuel_consumption_fms_high,
# MAGIC     std(fuel_consumption_fms_high) as std_fuel_consumption_fms_high
# MAGIC   from bms_ds_prod.bms_ds_dasc.temp_telematics_raw
# MAGIC   where date between "2025-10-01" and "2025-10-18"
# MAGIC   group by date
# MAGIC ) t
# MAGIC unpivot (
# MAGIC   metric_value for metric_name in (
# MAGIC     mean_fuel_consumption_ecol,
# MAGIC     std_fuel_consumption_ecol,
# MAGIC     mean_fuel_consumption_ecor,
# MAGIC     std_fuel_consumption_ecor,
# MAGIC     mean_fuel_consumption_fms_high,
# MAGIC     std_fuel_consumption_fms_high
# MAGIC   )
# MAGIC ))
# MAGIC
# MAGIC union all
# MAGIC
# MAGIC (select 
# MAGIC   date,
# MAGIC   "bronze" as pipeline_stage,
# MAGIC   metric_name,
# MAGIC   metric_value, 
# MAGIC   now() as created_at 
# MAGIC from (
# MAGIC   select 
# MAGIC     date,
# MAGIC     mean(fuel_consumption_ecol) as mean_fuel_consumption_ecol,
# MAGIC     std(fuel_consumption_ecol) as std_fuel_consumption_ecol,
# MAGIC     mean(fuel_consumption_ecor) as mean_fuel_consumption_ecor,
# MAGIC     std(fuel_consumption_ecor) as std_fuel_consumption_ecor,
# MAGIC     mean(fuel_consumption_fms_high) as mean_fuel_consumption_fms_high,
# MAGIC     std(fuel_consumption_fms_high) as std_fuel_consumption_fms_high
# MAGIC   from bms_ds_prod.bms_ds_dasc.temp_telematics_bronze
# MAGIC   where date between "2025-10-01" and "2025-10-18"
# MAGIC   group by date
# MAGIC ) t
# MAGIC unpivot (
# MAGIC   metric_value for metric_name in (
# MAGIC     mean_fuel_consumption_ecol,
# MAGIC     std_fuel_consumption_ecol,
# MAGIC     mean_fuel_consumption_ecor,
# MAGIC     std_fuel_consumption_ecor,
# MAGIC     mean_fuel_consumption_fms_high,
# MAGIC     std_fuel_consumption_fms_high
# MAGIC   )
# MAGIC )))
# MAGIC group by date, pipeline_stage

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC   date,
# MAGIC   "bronze" as pipeline_stage,
# MAGIC   metric_name,
# MAGIC   metric_value, 
# MAGIC   now() as created_at 
# MAGIC from (
# MAGIC   select 
# MAGIC     date,
# MAGIC     mean(fuel_consumption_ecol) as mean_fuel_consumption_ecol,
# MAGIC     std(fuel_consumption_ecol) as std_fuel_consumption_ecol,
# MAGIC     mean(fuel_consumption_ecor) as mean_fuel_consumption_ecor,
# MAGIC     std(fuel_consumption_ecor) as std_fuel_consumption_ecor,
# MAGIC     mean(fuel_consumption_fms_high) as mean_fuel_consumption_fms_high,
# MAGIC     std(fuel_consumption_fms_high) as std_fuel_consumption_fms_high
# MAGIC   from bms_ds_prod.bms_ds_dasc.temp_telematics_bronze
# MAGIC   group by date
# MAGIC ) t
# MAGIC unpivot (
# MAGIC   metric_value for metric_name in (
# MAGIC     mean_fuel_consumption_ecol,
# MAGIC     std_fuel_consumption_ecol,
# MAGIC     mean_fuel_consumption_ecor,
# MAGIC     std_fuel_consumption_ecor,
# MAGIC     mean_fuel_consumption_fms_high,
# MAGIC     std_fuel_consumption_fms_high
# MAGIC   )
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC select * 
# MAGIC from bms_ds_prod.bms_ds_dasc.temp_raw_metrics 
# MAGIC where date = '2025-10-20' --and pipeline_stage = 'bronze'

# COMMAND ----------

# %sql
# UPDATE bms_ds_prod.bms_ds_dasc.temp_raw_metrics
# SET metric_name = concat(pipeline_stage, '_', metric_name)
# WHERE metric_name IN (
#   'mean_fuel_consumption_ecol',
#   'std_fuel_consumption_ecol',
#   'mean_fuel_consumption_ecor',
#   'std_fuel_consumption_ecor',
#   'mean_fuel_consumption_fms_high',
#   'std_fuel_consumption_fms_high'
# )