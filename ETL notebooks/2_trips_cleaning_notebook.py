# Databricks notebook source
from datetime import datetime
from pyspark.sql import functions as F, Window, Row
from pyspark.sql import types as T

# COMMAND ----------

# dbutils.widgets.text("date", defaultValue="2025-12-10", label="Date")
# dbutils.widgets.text("table_name", defaultValue="bms_ds_bronze.trips", label="Table")
source_table = dbutils.widgets.get("table_name")
run_date = dbutils.widgets.get("date")

# COMMAND ----------

#Starting ingestion timer
ingestion_start_ts = datetime.utcnow()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading table 

# COMMAND ----------

# DBTITLE 1,Cell 8
def metrics_dict_to_df(metrics: dict, run_date: str, stage: str):
    rows = [
        Row(
            date=datetime.strptime(run_date, "%Y-%m-%d").date(),
            pipeline_stage=stage,
            metric_name=k,
            metric_value=float(v),
            created_at=datetime.utcnow()
        )
        for k, v in metrics.items()
    ]
    return spark.createDataFrame(rows)

def write_metrics_df(metrics_df):
    display(metrics_df)
    metrics_df.write.mode("append").saveAsTable(metrics_table)

# COMMAND ----------

# DBTITLE 1,Cell 7
def clean_raw_trips_with_metrics(
    df,
    run_date: str,
    min_duration_limit: int = 31,
    max_daily_events: int = 124,
    max_duration: float = 86400.0,
    max_distance_km: float = 1200.0,
):
    metrics = {}

    # ----------------------------
    # Initial volume
    # ----------------------------
    metrics["bronze_input_rows"] = df.count()

    # Dropping latitude and longitude columns
    df = df.drop("start_latitude", "start_longitude", "end_latitude", "end_longitude")

    # ----------------------------
    # Trip type filtering
    # ----------------------------
    metrics["bronze_correction_trips_removed"] = df.filter(F.col("trip_type") == 4).count()
    df = df.filter(F.col("trip_type") != 4)

    # ----------------------------
    # Primary key + location null validation (FIXED)
    # ----------------------------
    pk_null_condition = (
        F.col("unit_id").isNull() |
        F.col("start").isNull() |
        F.col("end").isNull()
    )

    metrics["bronze_null_primary_key_rows"] = df.filter(pk_null_condition).count()
    df = df.filter(~pk_null_condition)

    # ----------------------------
    # Deduplication (simple, safe)
    # ----------------------------
    before_dedup = df.count()
    df = df.dropDuplicates([
        "unit_id",
        "start",
        "end",
    ])
    metrics["bronze_duplicate_rows_removed"] = before_dedup - df.count()

    # ----------------------------
    # Start < End validation
    # ----------------------------
    metrics["bronze_start_after_end_rows"] = df.filter(F.col("start") >= F.col("end")).count()
    df = df.filter(F.col("start") < F.col("end"))

    # ----------------------------
    # Derivations + Distribution Tracking
    # ----------------------------
    df = (
        df.withColumn(
            "duration",
            (F.col("end").cast("double") - F.col("start").cast("double")).cast("int")
        )
        .withColumn("distance_km", F.col("distance") / 1000.0)
    )
    
    # Calculate distribution metrics for shift detection (single aggregation)
    post_derivation_stats = (
        df.select([
            F.mean("distance_km").alias("bronze_distance_km_mean"),
            F.stddev("distance_km").alias("bronze_distance_km_std"),
            F.mean("duration").alias("bronze_duration_mean"),
            F.stddev("duration").alias("bronze_duration_std"),
            F.sum(F.when(F.col("avg_speed") > 200, 1).otherwise(0)).alias("bronze_impossible_speed_events"),
            F.sum(F.when(F.col("fuel_consumption") < 0, 1).otherwise(0)).alias("bronze_negative_fuel_events"),
            F.sum(F.when((F.col("distance_km") == 0) & (F.col("fuel_consumption") > 0), 1).otherwise(0)).alias("bronze_zero_distance_fuel_events")
        ])
        .collect()[0]
        .asDict()
    )
    metrics.update(post_derivation_stats)

    # ----------------------------
    # Avg speed validation
    # ----------------------------
    df = df.withColumn(
        "avg_speed_calc",
        F.col("distance_km") / (F.col("duration") / 3600.0)
    )

    invalid_avg_speed_condition = (
        (F.col("avg_speed") <= 0) |
        (F.col("avg_speed") > 300) |
        (F.col("avg_speed") <= F.col("avg_speed_calc"))
    )

    metrics["bronze_invalid_avg_speed_rows"] = df.filter(invalid_avg_speed_condition).count()

    df = df.withColumn(
        "avg_speed",
        F.when(invalid_avg_speed_condition, F.lit(None))
         .otherwise(F.col("avg_speed"))
    ).drop("avg_speed_calc")

    # ----------------------------
    # Duration & distance rules
    # ----------------------------
    rule_condition = (
        (F.col("duration") > min_duration_limit) &
        (F.col("duration") < max_duration) &
        (F.col("distance_km") > 0) &
        (F.col("distance_km") < max_distance_km)
    )

    metrics["bronze_rows_before_rule_filter"] = df.count()
    df = df.filter(rule_condition)
    metrics["bronze_rows_after_rule_filter"] = df.count()
    metrics["bronze_rows_dropped_by_rules"] = (
        metrics["bronze_rows_before_rule_filter"] - metrics["bronze_rows_after_rule_filter"]
    )

    # ----------------------------
    # Daily events per unit
    # ----------------------------
    w = Window.partitionBy("unit_id", "date")
    df = df.withColumn("daily_events", F.count("*").over(w))

    metrics["bronze_excessive_daily_events_units"] = df.filter(
        F.col("daily_events") >= max_daily_events
    ).count()

    df = df.filter(F.col("daily_events") < max_daily_events).drop("daily_events")

    # ----------------------------
    # Idle time correction
    # ----------------------------
    idle_invalid = (
        (F.col("idle_time") < 0) |
        (F.col("idle_time") >= F.col("duration"))
    )

    metrics["bronze_idle_time_invalid_corrected"] = df.filter(idle_invalid).count()

    df = df.withColumn(
        "idle_time",
        F.when(idle_invalid, F.lit(None)).otherwise(F.col("idle_time"))
    )

    # ----------------------------
    # Final volume + Survival Rate
    # ----------------------------
    metrics["bronze_output_rows"] = df.count()
    
    # Calculate survival rate for causal analysis
    if metrics["bronze_input_rows"] > 0:
        metrics["bronze_survival_rate"] = metrics["bronze_output_rows"] / metrics["bronze_input_rows"]
    else:
        metrics["bronze_survival_rate"] = 0.0

    return df, metrics

# COMMAND ----------

# Defining config
raw_layer_table = "bms_ds_prod.bms_ds_dasc.temp_telematics_raw"      
bronze_table_name = "bms_ds_prod.bms_ds_dasc.temp_telematics_bronze"
metrics_table = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"

stage = "bronze"

assert run_date, "Date widget value is required and cannot be empty."

print(f"Reading data from {raw_layer_table} for date {run_date}.")

#reading raw table for the date of the current run
raw_df = spark.table(raw_layer_table).filter(F.col("date") == run_date)

clean_df, metrics_dic = clean_raw_trips_with_metrics(
    raw_df,
    run_date
)

clean_df.write.mode("append").format("delta").partitionBy("date", "hour").saveAsTable(bronze_table_name)

# COMMAND ----------

ingestion_end_ts = datetime.utcnow()
ingestion_duration_sec = (ingestion_end_ts - ingestion_start_ts).total_seconds()
metrics_dic["bronze_ingestion_duration_sec"] = ingestion_duration_sec

# COMMAND ----------

# MAGIC %md
# MAGIC ## Record Metrics

# COMMAND ----------

metrics_df = metrics_dict_to_df(metrics_dic, run_date, stage)
write_metrics_df(metrics_df)

# COMMAND ----------
