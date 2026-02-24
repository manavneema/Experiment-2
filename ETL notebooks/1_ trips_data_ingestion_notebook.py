# Databricks notebook source
# DBTITLE 1,Cell 1
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
import pandas as pd
from datetime import datetime

# COMMAND ----------

# dbutils.widgets.text("date", defaultValue="2025-12-15", label="Date")
# dbutils.widgets.text("table_name", defaultValue="bms_ds_bronze.trips", label="Table")

# COMMAND ----------

#Starting ingestion timer
ingestion_start_ts = datetime.utcnow()

# COMMAND ----------

# Defining config
raw_layer_table = "bms_ds_prod.bms_ds_dasc.temp_telematics_raw"      
metrics_table = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"
source_table = dbutils.widgets.get("table_name")       
run_date = dbutils.widgets.get("date") or '2023-01-01'

assert run_date, "Date widget value is required and cannot be empty."

print(source_table)
print(run_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading raw data

# COMMAND ----------

raw_trips_df = spark.sql(f"""
    select
        account_id,
        driver_id,
        unit_id,
        start_latitude,
        start_longitude,
        end_latitude,
        end_longitude,
        avg_speed,
        distance,
        start,
        end,
        trip_type,
        fuel_consumption,
        fuel_consumption_ecol,
        fuel_consumption_ecor,
        fuel_consumption_fms_high,
        fuel_consumption_fms_low,
        gps_coverage,
        idle_time,
        max_speed,
        date,
        hour
    from {source_table}
    where date = '{run_date}'
""")

# COMMAND ----------

record_count = raw_trips_df.count()
print(record_count)

null_counts = (
    raw_trips_df
    .select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in raw_trips_df.columns
    ])
    .collect()[0]
    .asDict()
)

min_start = raw_trips_df.select(F.min("start")).first()[0]
max_end   = raw_trips_df.select(F.max("end")).first()[0]

# Distribution metrics for causal analysis (single aggregation)
distribution_stats = (
    raw_trips_df
    .select([
        F.mean("distance").alias("distance_mean"),
        F.stddev("distance").alias("distance_std"),
        F.mean("avg_speed").alias("avg_speed_mean"), 
        F.stddev("avg_speed").alias("avg_speed_std"),
        F.mean("fuel_consumption").alias("fuel_consumption_mean"),
        F.stddev("fuel_consumption").alias("fuel_consumption_std"),
        F.countDistinct("unit_id").alias("unique_units"),
        F.sum(F.when(F.col("gps_coverage") < 0.8, 1).otherwise(0)).alias("poor_gps_coverage_count")
    ])
    .collect()[0]
    .asDict()
)

# print(null_counts)
# print(min_start, max_end)
# display(raw_trips_df)

# COMMAND ----------

#Writing to raw table
raw_trips_df.write.mode("append").format("delta").partitionBy("date", "hour").saveAsTable(raw_layer_table)

# COMMAND ----------

ingestion_end_ts = datetime.utcnow()
ingestion_duration_sec = (ingestion_end_ts - ingestion_start_ts).total_seconds()

# COMMAND ----------

#Wrote this DF to the table
display(raw_trips_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Cell 10
metrics_rows = []

# Convert run_date string to date object
run_date_obj = datetime.strptime(run_date, "%Y-%m-%d").date()
created_at_ts = datetime.utcnow()

metrics_rows.append((
    run_date_obj,
    "raw",
    "raw_input_record_count",
    float(record_count),
    created_at_ts
))

metrics_rows.append((
    run_date_obj,
    "raw",
    "raw_ingestion_duration_sec",
    float(ingestion_duration_sec),
    created_at_ts
))

metrics_rows.append((
    run_date_obj,
    "raw",
    "raw_min_trip_start_ts",
    None if min_start is None else float(min_start.timestamp()),
    created_at_ts
))

metrics_rows.append((
    run_date_obj,
    "raw",
    "raw_max_trip_end_ts",
    None if max_end is None else float(max_end.timestamp()),
    created_at_ts
))

# Add distribution metrics for causal analysis
for metric_name, metric_value in distribution_stats.items():
    if metric_value is not None:
        metrics_rows.append((
            run_date_obj,
            "raw",
            f"raw_{metric_name}",
            float(metric_value),
            created_at_ts
        ))

# Add temporal coverage metric
if min_start and max_end:
    temporal_coverage_hours = (max_end.timestamp() - min_start.timestamp()) / 3600
    metrics_rows.append((
        run_date_obj,
        "raw",
        "raw_temporal_coverage_hours",
        float(temporal_coverage_hours),
        created_at_ts
    ))

for col_name, null_count in null_counts.items():
    metrics_rows.append((
        run_date_obj,
        "raw",
        f"raw_null_count_{col_name}",
        float(null_count),
        created_at_ts
    ))

# print(metrics_rows)

# COMMAND ----------

# DBTITLE 1,Untitled
metrics_schema = T.StructType([
    T.StructField("date", T.DateType(), False),
    T.StructField("pipeline_stage", T.StringType(), False),
    T.StructField("metric_name", T.StringType(), False),
    T.StructField("metric_value", T.DoubleType(), True),
    T.StructField("created_at", T.TimestampType(), True)
])

metrics_df = spark.createDataFrame(metrics_rows, schema=metrics_schema)
print("Recorded following metrics for ingestion:")
display(metrics_df)

(
    metrics_df
    .write
    .mode("append")
    .format("delta")
    .saveAsTable(metrics_table)
)

# COMMAND ----------

print(f"RAW ingestion completed for date {run_date}")
print(f"Records ingested: {record_count}")
print(f"Ingestion duration (sec): {ingestion_duration_sec}")


# COMMAND ----------
