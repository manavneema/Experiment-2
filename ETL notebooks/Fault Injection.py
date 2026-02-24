# Databricks notebook source
# MAGIC %md
# MAGIC # Fault Injection Framework for Causal Discovery
# MAGIC 
# MAGIC This notebook injects controlled faults into ETL pipeline data to generate training samples for causal discovery.
# MAGIC 
# MAGIC **IMPORTANT:** This notebook only READS from `bms_ds_bronze.trips` and APPENDS to a temporary table.
# MAGIC It will NEVER modify `bms_ds_bronze.trips`.
# MAGIC 
# MAGIC ## Fault Categories (42 runs total):
# MAGIC 
# MAGIC | Category | Runs | Columns Affected | Downstream Impact |
# MAGIC |----------|------|------------------|-------------------|
# MAGIC | **null_primary_key** | 6 | unit_id, start, end | bronze_null_primary_key_rows → survival_rate |
# MAGIC | **null_numeric_features** | 6 | idle_time, avg_speed, gps_coverage | silver_avg_speed_imputed, KPI numerators |
# MAGIC | **duplicate_rows** | 4 | all columns | bronze_duplicate_rows_removed |
# MAGIC | **distribution_shift** | 8 | fuel, distance, avg_speed, idle_time | raw_*_mean → bronze_*_mean → KPIs & ML |
# MAGIC | **invalid_duration** | 4 | start, end (duration calc) | bronze_rows_dropped_by_rules → survival_rate |
# MAGIC | **timestamp_order_violation** | 4 | start, end | bronze_start_after_end_rows |
# MAGIC | **numeric_corruption** | 6 | distance, idle_time | KPI divide-by-zero, invalid ratios |
# MAGIC | **trip_type_corruption** | 4 | trip_type | bronze_correction_trips_removed |

# COMMAND ----------

from pyspark.sql.functions import (
    rand, when, col, lit, expr, 
    unix_timestamp, from_unixtime,
    abs as spark_abs
)
from pyspark.sql import DataFrame
from datetime import datetime
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC 
# MAGIC Define all fault injection test cases below. Each test case specifies:
# MAGIC - `date`: The date to pull data from bms_ds_bronze.trips
# MAGIC - `fault_type`: Type of fault to inject
# MAGIC - `severity`: Fraction of rows affected (0.0 to 1.0)
# MAGIC - `description`: Human-readable description for logging

# COMMAND ----------

# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

# Source table (READ ONLY - we never write to this)
SOURCE_TABLE = "bms_ds_bronze.trips"

# Target table (we APPEND fault-injected data here)
TARGET_TABLE = "bms_ds_prod.bms_ds_dasc.temp_fault_injection_training"

# Whether to create the target table fresh (True) or append to existing (False)
# Set to True only for the first run to initialize the table
CREATE_TARGET_TABLE_FRESH = False

# =============================================================================
# TEST CASE DEFINITIONS
# =============================================================================
# Each test case: {"date": "YYYY-MM-DD", "fault_type": "...", "severity": 0.X, "description": "..."}
#
# Fault types (8 categories, 42 runs):
#   - "null_primary_key"       : Nulls in unit_id/start/end (PK columns)
#   - "null_numeric_features"  : Nulls in idle_time/avg_speed/gps_coverage
#   - "duplicate_rows"         : Duplicates a fraction of rows
#   - "distribution_shift"     : Drift in fuel/distance/avg_speed/idle_time
#   - "invalid_duration"       : Creates invalid duration scenarios
#   - "timestamp_order_violation": Makes start >= end
#   - "numeric_corruption"     : Zeros/negatives in distance/idle_time
#   - "trip_type_corruption"   : Sets trip_type=4 (correction trips filtered)
# =============================================================================

FAULT_INJECTION_CONFIG = [
    # =========================================================================
    # FAULT TYPE 1: Null Primary Key (6 runs)
    # Tests: raw_null_count_unit_id/start/end → bronze_null_primary_key_rows → survival_rate
    # =========================================================================
    {"date": "2025-10-20", "fault_type": "null_primary_key", "severity": 0.10, "description": "10% null PK (unit_id focus)", "target_col": "unit_id"},
    {"date": "2025-10-21", "fault_type": "null_primary_key", "severity": 0.10, "description": "10% null PK (timestamps focus)", "target_col": "timestamps"},
    {"date": "2025-10-22", "fault_type": "null_primary_key", "severity": 0.25, "description": "25% null PK (mixed)", "target_col": "mixed"},
    {"date": "2025-10-23", "fault_type": "null_primary_key", "severity": 0.25, "description": "25% null PK (unit_id focus)", "target_col": "unit_id"},
    {"date": "2025-10-24", "fault_type": "null_primary_key", "severity": 0.40, "description": "40% null PK (timestamps focus)", "target_col": "timestamps"},
    {"date": "2025-10-25", "fault_type": "null_primary_key", "severity": 0.40, "description": "40% null PK (mixed)", "target_col": "mixed"},
    
    # =========================================================================
    # FAULT TYPE 2: Null Numeric Features (6 runs)
    # Tests: silver_avg_speed_imputed, idle_time KPI numerator, gps_coverage ML feature
    # =========================================================================
    {"date": "2025-10-26", "fault_type": "null_numeric_features", "severity": 0.15, "description": "15% null idle_time", "target_col": "idle_time"},
    {"date": "2025-10-27", "fault_type": "null_numeric_features", "severity": 0.15, "description": "15% null avg_speed", "target_col": "avg_speed"},
    {"date": "2025-10-28", "fault_type": "null_numeric_features", "severity": 0.30, "description": "30% null gps_coverage", "target_col": "gps_coverage"},
    {"date": "2025-10-29", "fault_type": "null_numeric_features", "severity": 0.30, "description": "30% null idle_time + avg_speed", "target_col": "idle_speed"},
    {"date": "2025-10-30", "fault_type": "null_numeric_features", "severity": 0.50, "description": "50% null max_speed", "target_col": "max_speed"},
    {"date": "2025-10-31", "fault_type": "null_numeric_features", "severity": 0.50, "description": "50% null mixed numerics", "target_col": "mixed"},
    
    # =========================================================================
    # FAULT TYPE 3: Duplicate Rows (4 runs)
    # Tests: bronze_duplicate_rows_removed → bronze_output_rows
    # =========================================================================
    {"date": "2025-11-01", "fault_type": "duplicate_rows", "severity": 0.10, "description": "10% duplicate rows"},
    {"date": "2025-11-02", "fault_type": "duplicate_rows", "severity": 0.25, "description": "25% duplicate rows"},
    {"date": "2025-11-03", "fault_type": "duplicate_rows", "severity": 0.40, "description": "40% duplicate rows"},
    {"date": "2025-11-04", "fault_type": "duplicate_rows", "severity": 0.60, "description": "60% duplicate rows - EXTREME"},
    
    # =========================================================================
    # FAULT TYPE 4: Distribution Shift (8 runs)
    # Tests: raw_*_mean → bronze_*_mean → KPIs (fuel_per_100km, idling_per_100km) & ML
    # =========================================================================
    {"date": "2025-11-05", "fault_type": "distribution_shift", "severity": 0.20, "description": "20% fuel drift (+)", "target_col": "fuel_consumption", "direction": "up"},
    {"date": "2025-11-06", "fault_type": "distribution_shift", "severity": 0.20, "description": "20% distance drift (+)", "target_col": "distance", "direction": "up"},
    {"date": "2025-11-07", "fault_type": "distribution_shift", "severity": 0.30, "description": "30% avg_speed drift (+)", "target_col": "avg_speed", "direction": "up"},
    {"date": "2025-11-08", "fault_type": "distribution_shift", "severity": 0.30, "description": "30% idle_time drift (+)", "target_col": "idle_time", "direction": "up"},
    {"date": "2025-11-09", "fault_type": "distribution_shift", "severity": 0.40, "description": "40% fuel drift (-)", "target_col": "fuel_consumption", "direction": "down"},
    {"date": "2025-11-10", "fault_type": "distribution_shift", "severity": 0.40, "description": "40% distance drift (-)", "target_col": "distance", "direction": "down"},
    {"date": "2025-11-11", "fault_type": "distribution_shift", "severity": 0.50, "description": "50% multi-col drift (fuel+distance)", "target_col": "fuel_distance", "direction": "up"},
    {"date": "2025-11-12", "fault_type": "distribution_shift", "severity": 0.50, "description": "50% multi-col drift (speed+idle)", "target_col": "speed_idle", "direction": "up"},
    
    # =========================================================================
    # FAULT TYPE 5: Invalid Duration (4 runs)
    # Tests: bronze_rows_dropped_by_rules → bronze_survival_rate
    # =========================================================================
    {"date": "2025-11-13", "fault_type": "invalid_duration", "severity": 0.15, "description": "15% too-short trips (<31s)"},
    {"date": "2025-11-14", "fault_type": "invalid_duration", "severity": 0.25, "description": "25% too-short trips (<31s)"},
    {"date": "2025-11-15", "fault_type": "invalid_duration", "severity": 0.35, "description": "35% too-long trips (>24h)"},
    {"date": "2025-11-16", "fault_type": "invalid_duration", "severity": 0.45, "description": "45% mixed invalid duration"},
    
    # =========================================================================
    # FAULT TYPE 6: Timestamp Order Violation (4 runs)
    # Tests: bronze_start_after_end_rows → bronze_output_rows
    # =========================================================================
    {"date": "2025-11-17", "fault_type": "timestamp_order_violation", "severity": 0.10, "description": "10% start >= end"},
    {"date": "2025-11-18", "fault_type": "timestamp_order_violation", "severity": 0.25, "description": "25% start >= end"},
    {"date": "2025-11-19", "fault_type": "timestamp_order_violation", "severity": 0.40, "description": "40% start >= end"},
    {"date": "2025-11-20", "fault_type": "timestamp_order_violation", "severity": 0.55, "description": "55% start >= end - EXTREME"},
    
    # =========================================================================
    # FAULT TYPE 7: Numeric Corruption (6 runs)
    # Tests: KPI divide-by-zero (distance=0), negative values, invalid ratios
    # =========================================================================
    {"date": "2025-11-21", "fault_type": "numeric_corruption", "severity": 0.15, "description": "15% zero distance", "target_col": "distance", "corruption": "zero"},
    {"date": "2025-11-22", "fault_type": "numeric_corruption", "severity": 0.15, "description": "15% negative idle_time", "target_col": "idle_time", "corruption": "negative"},
    {"date": "2025-11-23", "fault_type": "numeric_corruption", "severity": 0.30, "description": "30% extreme distance (>1200km)", "target_col": "distance", "corruption": "extreme"},
    {"date": "2025-11-24", "fault_type": "numeric_corruption", "severity": 0.30, "description": "30% negative fuel", "target_col": "fuel_consumption", "corruption": "negative"},
    {"date": "2025-11-25", "fault_type": "numeric_corruption", "severity": 0.45, "description": "45% mixed corruption (distance+idle)", "target_col": "mixed", "corruption": "mixed"},
    {"date": "2025-11-26", "fault_type": "numeric_corruption", "severity": 0.45, "description": "45% impossible speed (>300 km/h)", "target_col": "avg_speed", "corruption": "extreme"},
    
    # =========================================================================
    # FAULT TYPE 8: Trip Type Corruption (4 runs)
    # Tests: bronze_correction_trips_removed → bronze_output_rows
    # =========================================================================
    {"date": "2025-11-27", "fault_type": "trip_type_corruption", "severity": 0.15, "description": "15% trip_type=4 (correction)"},
    {"date": "2025-11-28", "fault_type": "trip_type_corruption", "severity": 0.30, "description": "30% trip_type=4 (correction)"},
    {"date": "2025-11-29", "fault_type": "trip_type_corruption", "severity": 0.45, "description": "45% trip_type=4 (correction)"},
    {"date": "2025-11-30", "fault_type": "trip_type_corruption", "severity": 0.60, "description": "60% trip_type=4 - EXTREME"},
]

print(f"Total fault injection test cases configured: {len(FAULT_INJECTION_CONFIG)}")
print(f"Source table: {SOURCE_TABLE} (READ ONLY)")
print(f"Target table: {TARGET_TABLE} (APPEND)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fault Injection Functions
# MAGIC 
# MAGIC Each function takes a DataFrame and severity, returns the modified DataFrame.

# COMMAND ----------

def inject_null_primary_key(df: DataFrame, severity: float, seed: int = None, target_col: str = "mixed") -> DataFrame:
    """
    Inject nulls into primary key columns (unit_id, start, end).
    
    target_col options:
        - "unit_id": Only null unit_id
        - "timestamps": Only null start/end
        - "mixed": Randomly null any of the three
    
    Causal path tested: raw_null_count_* → bronze_null_primary_key_rows → survival_rate
    """
    rand_col = rand(seed) if seed else rand()
    which_col = rand()  # For mixed mode
    
    if target_col == "unit_id":
        return df.withColumn(
            "unit_id",
            when(rand_col < severity, lit(None)).otherwise(col("unit_id"))
        )
    elif target_col == "timestamps":
        df = df.withColumn("_fault_rand", rand_col)
        df = df.withColumn("_which", rand())
        df = df.withColumn(
            "start",
            when((col("_fault_rand") < severity) & (col("_which") < 0.5), lit(None)).otherwise(col("start"))
        ).withColumn(
            "end",
            when((col("_fault_rand") < severity) & (col("_which") >= 0.5), lit(None)).otherwise(col("end"))
        )
        return df.drop("_fault_rand", "_which")
    else:  # mixed
        df = df.withColumn("_fault_rand", rand_col)
        df = df.withColumn("_which", which_col)
        df = df.withColumn(
            "unit_id",
            when((col("_fault_rand") < severity) & (col("_which") < 0.33), lit(None)).otherwise(col("unit_id"))
        ).withColumn(
            "start",
            when((col("_fault_rand") < severity) & (col("_which") >= 0.33) & (col("_which") < 0.66), lit(None)).otherwise(col("start"))
        ).withColumn(
            "end",
            when((col("_fault_rand") < severity) & (col("_which") >= 0.66), lit(None)).otherwise(col("end"))
        )
        return df.drop("_fault_rand", "_which")


def inject_null_numeric_features(df: DataFrame, severity: float, seed: int = None, target_col: str = "mixed") -> DataFrame:
    """
    Inject nulls into numeric feature columns used in KPIs and ML.
    
    target_col options:
        - "idle_time": KPI numerator for idling_per_100km
        - "avg_speed": ML feature, triggers silver_avg_speed_imputed
        - "gps_coverage": ML feature
        - "max_speed": ML feature
        - "idle_speed": Both idle_time and avg_speed
        - "mixed": Random selection
    
    Causal path tested: silver_avg_speed_imputed, KPI numerator issues
    """
    rand_col = rand(seed) if seed else rand()
    which_col = rand()
    
    target_cols = {
        "idle_time": ["idle_time"],
        "avg_speed": ["avg_speed"],
        "gps_coverage": ["gps_coverage"],
        "max_speed": ["max_speed"],
        "idle_speed": ["idle_time", "avg_speed"],
        "mixed": ["idle_time", "avg_speed", "gps_coverage", "max_speed"]
    }
    
    cols_to_null = target_cols.get(target_col, ["idle_time"])
    
    df = df.withColumn("_fault_rand", rand_col)
    df = df.withColumn("_which", which_col)
    
    for i, c in enumerate(cols_to_null):
        if len(cols_to_null) == 1:
            df = df.withColumn(c, when(col("_fault_rand") < severity, lit(None)).otherwise(col(c)))
        else:
            lower = i / len(cols_to_null)
            upper = (i + 1) / len(cols_to_null)
            df = df.withColumn(
                c,
                when(
                    (col("_fault_rand") < severity) & (col("_which") >= lower) & (col("_which") < upper),
                    lit(None)
                ).otherwise(col(c))
            )
    
    return df.drop("_fault_rand", "_which")


def inject_duplicate_rows(df: DataFrame, severity: float, seed: int = 42, **kwargs) -> DataFrame:
    """
    Duplicate a fraction of rows and append them back.
    
    Causal path tested: bronze_duplicate_rows_removed → bronze_output_rows
    """
    sampled_df = df.sample(fraction=severity, seed=seed)
    return df.unionByName(sampled_df)


def inject_distribution_shift(df: DataFrame, severity: float, seed: int = None, 
                               target_col: str = "fuel_consumption", direction: str = "up") -> DataFrame:
    """
    Apply multiplicative drift to numeric columns to simulate sensor drift or data corruption.
    
    target_col options:
        - "fuel_consumption": KPI numerator
        - "distance": KPI denominator
        - "avg_speed": ML feature
        - "idle_time": KPI numerator
        - "fuel_distance": Both fuel and distance
        - "speed_idle": Both speed and idle
    
    direction: "up" (multiply by 1+severity) or "down" (multiply by 1-severity)
    
    Causal path tested: raw_*_mean → bronze_*_mean → KPIs & ML predictions
    """
    rand_col = rand(seed) if seed else rand()
    drift_factor = (1 + severity) if direction == "up" else (1 - severity * 0.5)  # down is less aggressive
    
    target_cols = {
        "fuel_consumption": ["fuel_consumption"],
        "distance": ["distance"],
        "avg_speed": ["avg_speed"],
        "idle_time": ["idle_time"],
        "fuel_distance": ["fuel_consumption", "distance"],
        "speed_idle": ["avg_speed", "idle_time"]
    }
    
    cols_to_shift = target_cols.get(target_col, ["fuel_consumption"])
    
    for c in cols_to_shift:
        # Cast back to long to preserve original schema (columns are long type in Delta table)
        df = df.withColumn(
            c,
            when(rand_col < severity, (col(c) * drift_factor).cast("long")).otherwise(col(c))
        )
    
    return df


def inject_invalid_duration(df: DataFrame, severity: float, seed: int = None, **kwargs) -> DataFrame:
    """
    Create invalid duration scenarios:
    - Too short (<31 seconds) → filtered by min_duration_limit
    - Too long (>86400 seconds / 24h) → filtered by max_duration
    
    Causal path tested: bronze_rows_dropped_by_rules → bronze_survival_rate
    """
    rand_col = rand(seed) if seed else rand()
    duration_type = rand()  # Determines short vs long
    
    # 70% too short, 30% too long
    return df.withColumn(
        "end",
        when(
            rand_col < severity,
            when(
                duration_type < 0.7,
                col("start") + expr("INTERVAL 10 SECONDS")  # Too short
            ).otherwise(
                col("start") + expr("INTERVAL 25 HOURS")  # Too long
            )
        ).otherwise(col("end"))
    )


def inject_timestamp_order_violation(df: DataFrame, severity: float, seed: int = None, **kwargs) -> DataFrame:
    """
    Make start >= end for a fraction of rows by swapping timestamps.
    
    Causal path tested: bronze_start_after_end_rows → bronze_output_rows
    """
    rand_col = rand(seed) if seed else rand()
    
    return df.withColumn(
        "_temp_start", col("start")
    ).withColumn(
        "start",
        when(rand_col < severity, col("end")).otherwise(col("start"))
    ).withColumn(
        "end",
        when(rand_col < severity, col("_temp_start")).otherwise(col("end"))
    ).drop("_temp_start")


def inject_numeric_corruption(df: DataFrame, severity: float, seed: int = None,
                               target_col: str = "distance", corruption: str = "zero") -> DataFrame:
    """
    Corrupt numeric values with zeros, negatives, or extreme values.
    
    target_col: "distance", "idle_time", "fuel_consumption", "avg_speed", "mixed"
    corruption: "zero", "negative", "extreme", "mixed"
    
    Causal paths tested:
        - distance=0 → KPI divide-by-zero
        - negative fuel → bronze_negative_fuel_events
        - extreme speed → bronze_impossible_speed_events
        - idle >= duration → bronze_idle_time_invalid_corrected
    """
    rand_col = rand(seed) if seed else rand()
    corrupt_type = rand()
    
    # Define corruption values
    corruption_values = {
        "distance": {"zero": 0, "negative": -1000, "extreme": 2000000},  # 2000km in meters
        "idle_time": {"zero": 0, "negative": -100, "extreme": 100000},  # > typical duration
        "fuel_consumption": {"zero": 0, "negative": -50, "extreme": 10000},
        "avg_speed": {"zero": 0, "negative": -50, "extreme": 500}  # 500 km/h impossible
    }
    
    if target_col == "mixed":
        # Apply different corruptions to different columns
        df = df.withColumn("_fault_rand", rand_col)
        df = df.withColumn("_which", corrupt_type)
        df = df.withColumn(
            "distance",
            when((col("_fault_rand") < severity) & (col("_which") < 0.5), lit(0)).otherwise(col("distance"))
        ).withColumn(
            "idle_time",
            when((col("_fault_rand") < severity) & (col("_which") >= 0.5), lit(-100)).otherwise(col("idle_time"))
        )
        return df.drop("_fault_rand", "_which")
    
    col_corruptions = corruption_values.get(target_col, corruption_values["distance"])
    
    if corruption == "mixed":
        return df.withColumn(
            target_col,
            when(
                rand_col < severity,
                when(corrupt_type < 0.33, lit(col_corruptions["zero"]))
                .when(corrupt_type < 0.66, lit(col_corruptions["negative"]))
                .otherwise(lit(col_corruptions["extreme"]))
            ).otherwise(col(target_col))
        )
    else:
        corrupt_val = col_corruptions.get(corruption, 0)
        return df.withColumn(
            target_col,
            when(rand_col < severity, lit(corrupt_val)).otherwise(col(target_col))
        )


def inject_trip_type_corruption(df: DataFrame, severity: float, seed: int = None, **kwargs) -> DataFrame:
    """
    Set trip_type to 4 (correction trips) which are filtered out in bronze.
    
    Causal path tested: bronze_correction_trips_removed → bronze_output_rows
    """
    rand_col = rand(seed) if seed else rand()
    
    # Cast to byte to match table schema (trip_type is byte in Delta table)
    return df.withColumn(
        "trip_type",
        when(rand_col < severity, lit(4).cast("byte")).otherwise(col("trip_type"))
    )


# Map fault types to functions
FAULT_INJECTORS = {
    "null_primary_key": inject_null_primary_key,
    "null_numeric_features": inject_null_numeric_features,
    "duplicate_rows": inject_duplicate_rows,
    "distribution_shift": inject_distribution_shift,
    "invalid_duration": inject_invalid_duration,
    "timestamp_order_violation": inject_timestamp_order_violation,
    "numeric_corruption": inject_numeric_corruption,
    "trip_type_corruption": inject_trip_type_corruption,
}

print(f"Available fault types: {list(FAULT_INJECTORS.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Target Table (Run Once)
# MAGIC 
# MAGIC This cell creates the target table if `CREATE_TARGET_TABLE_FRESH = True`.
# MAGIC Only run this once when setting up the framework.

# COMMAND ----------

if CREATE_TARGET_TABLE_FRESH:
    print(f"⚠️ Creating fresh target table: {TARGET_TABLE}")
    
    # Read schema from source table
    schema_df = spark.sql(f"SELECT * FROM {SOURCE_TABLE} LIMIT 0")
    
    # Create empty table with same schema
    schema_df.write.mode("overwrite").format("delta").saveAsTable(TARGET_TABLE)
    
    print(f"✅ Created empty table: {TARGET_TABLE}")
else:
    print(f"ℹ️ Skipping table creation. Will append to existing: {TARGET_TABLE}")
    
    # Verify table exists
    try:
        count = spark.sql(f"SELECT COUNT(*) as cnt FROM {TARGET_TABLE}").collect()[0]["cnt"]
        print(f"✅ Target table exists with {count:,} rows")
    except Exception as e:
        print(f"❌ Target table does not exist! Set CREATE_TARGET_TABLE_FRESH = True")
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Fault Injection
# MAGIC 
# MAGIC Process each test case: read source data → inject fault → append to target table.

# COMMAND ----------

# Track results
injection_results = []
failed_cases = []

print("=" * 80)
print("STARTING FAULT INJECTION PIPELINE")
print("=" * 80)
print(f"Source: {SOURCE_TABLE} (READ ONLY)")
print(f"Target: {TARGET_TABLE} (APPEND)")
print(f"Total cases: {len(FAULT_INJECTION_CONFIG)}")
print("=" * 80)

for i, config in enumerate(FAULT_INJECTION_CONFIG):
    date = config["date"]
    fault_type = config["fault_type"]
    severity = config["severity"]
    description = config["description"]
    
    print(f"\n[{i+1}/{len(FAULT_INJECTION_CONFIG)}] Processing: {description}")
    print(f"    Date: {date}, Fault: {fault_type}, Severity: {severity:.0%}")
    
    try:
        # Step 1: Read source data for the specified date (READ ONLY)
        df = spark.sql(f"SELECT * FROM {SOURCE_TABLE} WHERE date = '{date}'")
        original_count = df.count()
        
        if original_count == 0:
            print(f"    ⚠️ WARNING: No data found for date {date}, skipping...")
            failed_cases.append({
                "config": config,
                "error": "No data found for date"
            })
            continue
        
        print(f"    📖 Read {original_count:,} rows from source")
        
        # Step 2: Apply fault injection with optional parameters
        injector_fn = FAULT_INJECTORS.get(fault_type)
        if not injector_fn:
            raise ValueError(f"Unknown fault type: {fault_type}")
        
        # Extract optional parameters from config
        extra_params = {k: v for k, v in config.items() if k not in ["date", "fault_type", "severity", "description"]}
        
        df_with_fault = injector_fn(df, severity, seed=i, **extra_params)  # Use index as seed for reproducibility
        fault_count = df_with_fault.count()
        
        print(f"    💉 Injected fault: {fault_type} at {severity:.0%} severity")
        print(f"    📊 Result: {fault_count:,} rows")
        
        # Step 3: Append to target table (NEVER overwrites)
        df_with_fault.write.mode("append").format("delta").saveAsTable(TARGET_TABLE)
        
        print(f"    ✅ Appended to {TARGET_TABLE}")
        
        # Track result
        injection_results.append({
            "date": date,
            "fault_type": fault_type,
            "severity": severity,
            "description": description,
            "original_count": original_count,
            "result_count": fault_count,
            "status": "SUCCESS"
        })
        
    except Exception as e:
        print(f"    ❌ ERROR: {str(e)}")
        failed_cases.append({
            "config": config,
            "error": str(e)
        })
        injection_results.append({
            "date": date,
            "fault_type": fault_type,
            "severity": severity,
            "description": description,
            "original_count": 0,
            "result_count": 0,
            "status": f"FAILED: {str(e)}"
        })

print("\n" + "=" * 80)
print("FAULT INJECTION COMPLETE")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Report

# COMMAND ----------

# Create summary DataFrame
results_df = spark.createDataFrame(injection_results)
display(results_df)

# COMMAND ----------

# Summary statistics
successful = len([r for r in injection_results if r["status"] == "SUCCESS"])
failed = len(failed_cases)

print("=" * 60)
print("INJECTION SUMMARY")
print("=" * 60)
print(f"Total test cases:  {len(FAULT_INJECTION_CONFIG)}")
print(f"Successful:        {successful}")
print(f"Failed:            {failed}")
print("=" * 60)

# Count by fault type
print("\nBy Fault Type:")
fault_type_counts = {}
for r in injection_results:
    ft = r["fault_type"]
    if ft not in fault_type_counts:
        fault_type_counts[ft] = {"success": 0, "failed": 0}
    if r["status"] == "SUCCESS":
        fault_type_counts[ft]["success"] += 1
    else:
        fault_type_counts[ft]["failed"] += 1

for ft, counts in fault_type_counts.items():
    print(f"  {ft}: {counts['success']} success, {counts['failed']} failed")

# COMMAND ----------

# Verify target table
print(f"\nTarget table verification: {TARGET_TABLE}")
final_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {TARGET_TABLE}").collect()[0]["cnt"]
print(f"Total rows in target table: {final_count:,}")

# Show date distribution
print("\nRows per date in target table:")
date_dist = spark.sql(f"""
    SELECT date, COUNT(*) as row_count 
    FROM {TARGET_TABLE} 
    GROUP BY date 
    ORDER BY date
""")
display(date_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Failed Cases (if any)

# COMMAND ----------

if failed_cases:
    print("❌ FAILED CASES:")
    for fc in failed_cases:
        print(f"  - {fc['config']['description']}: {fc['error']}")
else:
    print("✅ All cases completed successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC After running this notebook:
# MAGIC 
# MAGIC 1. **Run ETL Pipeline** for each date in the config:
# MAGIC    ```
# MAGIC    For each date in FAULT_INJECTION_CONFIG:
# MAGIC        - Set widget: date = config["date"]
# MAGIC        - Set widget: table_name = TARGET_TABLE  
# MAGIC        - Run: 1_trips_data_ingestion_notebook (skip - data already ingested)
# MAGIC        - Run: 2_trips_cleaning_notebook with source = TARGET_TABLE
# MAGIC        - Run: 3_trips_transformation_notebook
# MAGIC    ```
# MAGIC 
# MAGIC 2. **Collect Metrics** from `bms_ds_prod.bms_ds_dasc.temp_raw_metrics` table
# MAGIC 
# MAGIC 3. **Run Causal Discovery** on combined baseline + fault metrics

# COMMAND ----------

# Export config for ETL pipeline automation
print("\n📋 Dates to process through ETL pipeline:")
print("-" * 40)
for config in FAULT_INJECTION_CONFIG:
    print(f"  {config['date']} | {config['fault_type']} | {config['severity']:.0%}")

# Save config as JSON for automation
config_json = json.dumps(FAULT_INJECTION_CONFIG, indent=2)
print(f"\n📄 Config JSON (for automation):")
print(config_json[:500] + "..." if len(config_json) > 500 else config_json)

