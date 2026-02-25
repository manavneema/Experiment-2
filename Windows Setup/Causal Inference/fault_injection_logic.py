# Databricks notebook source
# MAGIC %md
# MAGIC # Fault Injection Logic for RCA Evaluation
# MAGIC 
# MAGIC This file contains functions to inject specific faults into raw trip DataFrames.
# MAGIC Each function takes a clean DataFrame and returns a faulty version that will
# MAGIC propagate through the ETL pipeline to produce anomalous metrics.
# MAGIC 
# MAGIC **Usage:**
# MAGIC ```python
# MAGIC from fault_injection_logic import inject_unit_id_nulls
# MAGIC faulty_df = inject_unit_id_nulls(clean_df, null_rate=0.4)
# MAGIC # Pass faulty_df to ETL pipeline
# MAGIC ```

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from typing import Union, Optional

# COMMAND ----------

# MAGIC %md
# MAGIC ## Case 1: Unit ID NULL Injection
# MAGIC **Root Causes:** `raw_null_count_unit_id`, `raw_unique_units`
# MAGIC 
# MAGIC Simulates data quality issue where unit_id (primary key) has NULL values.

# COMMAND ----------

def inject_unit_id_nulls(
    df: Union[pd.DataFrame, SparkDataFrame], 
    null_rate: float = 0.4,
    seed: int = 42
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Inject NULL values into unit_id column.
    
    Args:
        df: Input DataFrame with trips data
        null_rate: Fraction of rows to set unit_id to NULL (default 0.4 = 40%)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with injected NULL unit_id values
        
    Expected downstream effects:
        - raw_null_count_unit_id: Increases proportionally to null_rate
        - raw_unique_units: Decreases (fewer valid unit_ids)
        - bronze_null_primary_key_rows: Increases (rows flagged for missing PK)
    """
    if isinstance(df, pd.DataFrame):
        np.random.seed(seed)
        faulty_df = df.copy()
        n_nulls = int(len(faulty_df) * null_rate)
        null_indices = np.random.choice(faulty_df.index, size=n_nulls, replace=False)
        faulty_df.loc[null_indices, 'unit_id'] = None
        print(f"Injected NULL into {n_nulls}/{len(df)} unit_id values ({null_rate*100:.0f}%)")
        return faulty_df
    else:
        # Spark DataFrame
        faulty_df = df.withColumn(
            "unit_id",
            F.when(F.rand(seed) < null_rate, F.lit(None)).otherwise(F.col("unit_id"))
        )
        return faulty_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Case 2: Distance/GPS NULL Injection
# MAGIC **Root Causes:** `raw_null_count_distance`, `raw_poor_gps_coverage_count`
# MAGIC 
# MAGIC Simulates GPS sensor failures causing missing distance measurements.

# COMMAND ----------

def inject_distance_gps_nulls(
    df: Union[pd.DataFrame, SparkDataFrame],
    distance_null_rate: float = 0.35,
    gps_coverage_threshold: float = 0.5,
    seed: int = 42
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Inject NULL values into distance and degrade GPS coverage.
    
    Args:
        df: Input DataFrame with trips data
        distance_null_rate: Fraction of rows to set distance to NULL
        gps_coverage_threshold: Below this value, trip is flagged as poor GPS
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with injected distance NULLs and poor GPS coverage
        
    Expected downstream effects:
        - raw_null_count_distance: Increases
        - raw_poor_gps_coverage_count: Increases
        - bronze_distance_km_mean: Decreases (missing data)
        - p95_fuel_per_100km: Affected (fuel efficiency calculation)
    """
    if isinstance(df, pd.DataFrame):
        np.random.seed(seed)
        faulty_df = df.copy()
        
        # Inject distance NULLs
        n_distance_nulls = int(len(faulty_df) * distance_null_rate)
        distance_null_idx = np.random.choice(faulty_df.index, size=n_distance_nulls, replace=False)
        faulty_df.loc[distance_null_idx, 'distance'] = None
        
        # Also set related GPS columns to indicate poor coverage
        # Assuming columns like 'gps_coverage' or 'start_latitude/longitude'
        if 'gps_coverage' in faulty_df.columns:
            faulty_df.loc[distance_null_idx, 'gps_coverage'] = \
                np.random.uniform(0, gps_coverage_threshold, size=n_distance_nulls)
        
        # Also corrupt start/end coordinates for affected trips
        if 'start_longitude' in faulty_df.columns:
            coord_null_idx = np.random.choice(distance_null_idx, 
                                              size=int(len(distance_null_idx) * 0.5), 
                                              replace=False)
            faulty_df.loc[coord_null_idx, 'start_longitude'] = None
            faulty_df.loc[coord_null_idx, 'start_latitude'] = None
        
        print(f"Injected NULL into {n_distance_nulls}/{len(df)} distance values ({distance_null_rate*100:.0f}%)")
        return faulty_df
    else:
        # Spark DataFrame
        faulty_df = df.withColumn(
            "distance",
            F.when(F.rand(seed) < distance_null_rate, F.lit(None)).otherwise(F.col("distance"))
        )
        # Also corrupt GPS coordinates
        faulty_df = faulty_df.withColumn(
            "start_longitude",
            F.when(F.rand(seed + 1) < distance_null_rate * 0.5, F.lit(None)).otherwise(F.col("start_longitude"))
        ).withColumn(
            "start_latitude", 
            F.when(F.rand(seed + 2) < distance_null_rate * 0.5, F.lit(None)).otherwise(F.col("start_latitude"))
        )
        return faulty_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Case 3: Invalid Speed Values
# MAGIC **Root Causes:** `raw_avg_speed_mean`, `raw_null_count_avg_speed`
# MAGIC 
# MAGIC Simulates speed sensor malfunction producing invalid readings.

# COMMAND ----------

def inject_invalid_speeds(
    df: Union[pd.DataFrame, SparkDataFrame],
    invalid_rate: float = 0.25,
    null_rate: float = 0.15,
    speed_multiplier: float = 3.0,
    seed: int = 42
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Inject invalid speed values (abnormally high) and NULLs.
    
    Args:
        df: Input DataFrame with trips data
        invalid_rate: Fraction of rows to corrupt with high speed values
        null_rate: Fraction of rows to set speed to NULL
        speed_multiplier: Multiply valid speeds by this factor for invalid rows
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with corrupted avg_speed values
        
    Expected downstream effects:
        - raw_avg_speed_mean: Increases significantly
        - raw_null_count_avg_speed: Increases
        - bronze_invalid_avg_speed_rows: Increases (validation catches these)
        - silver_avg_speed_imputed: Affected by imputation logic
    """
    if isinstance(df, pd.DataFrame):
        np.random.seed(seed)
        faulty_df = df.copy()
        
        # Inject NULL speeds
        n_nulls = int(len(faulty_df) * null_rate)
        null_indices = np.random.choice(faulty_df.index, size=n_nulls, replace=False)
        faulty_df.loc[null_indices, 'avg_speed'] = None
        
        # Inject invalid high speeds (on non-null rows)
        remaining_idx = faulty_df.index.difference(null_indices)
        n_invalid = int(len(remaining_idx) * invalid_rate)
        invalid_indices = np.random.choice(remaining_idx, size=n_invalid, replace=False)
        
        # Multiply speed by factor (simulating sensor reading 3x actual)
        faulty_df.loc[invalid_indices, 'avg_speed'] = \
            faulty_df.loc[invalid_indices, 'avg_speed'] * speed_multiplier
        
        print(f"Injected {n_nulls} NULL speeds and {n_invalid} invalid high speeds")
        return faulty_df
    else:
        # Spark DataFrame
        faulty_df = df.withColumn(
            "avg_speed",
            F.when(F.rand(seed) < null_rate, F.lit(None))
             .when(F.rand(seed + 1) < invalid_rate, F.col("avg_speed") * speed_multiplier)
             .otherwise(F.col("avg_speed"))
        )
        return faulty_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Case 4: Duration/Idle Time Corruption
# MAGIC **Root Causes:** `bronze_rows_dropped_by_rules`, `bronze_survival_rate`
# MAGIC 
# MAGIC Simulates timestamp issues causing invalid duration calculations.

# COMMAND ----------

def inject_duration_anomalies(
    df: Union[pd.DataFrame, SparkDataFrame],
    negative_duration_rate: float = 0.10,
    extreme_duration_rate: float = 0.15,
    extreme_multiplier: float = 100.0,
    seed: int = 42
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Inject duration anomalies that will trigger validation rule drops.
    
    Args:
        df: Input DataFrame with trips data
        negative_duration_rate: Fraction of rows with negative/zero duration
        extreme_duration_rate: Fraction of rows with extremely long durations
        extreme_multiplier: Multiply duration for extreme cases
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with corrupted duration values
        
    Expected downstream effects:
        - bronze_rows_dropped_by_rules: Increases (validation rejects bad durations)
        - bronze_survival_rate: Decreases
        - bronze_duration_mean: May increase if extreme values pass validation
    """
    if isinstance(df, pd.DataFrame):
        np.random.seed(seed)
        faulty_df = df.copy()
        
        # Inject negative/zero durations (end_ts <= start_ts)
        n_negative = int(len(faulty_df) * negative_duration_rate)
        neg_indices = np.random.choice(faulty_df.index, size=n_negative, replace=False)
        
        if 'duration' in faulty_df.columns:
            faulty_df.loc[neg_indices, 'duration'] = -1 * abs(faulty_df.loc[neg_indices, 'duration'])
        elif 'trip_end_ts' in faulty_df.columns and 'trip_start_ts' in faulty_df.columns:
            # Swap start and end timestamps
            temp = faulty_df.loc[neg_indices, 'trip_start_ts'].copy()
            faulty_df.loc[neg_indices, 'trip_start_ts'] = faulty_df.loc[neg_indices, 'trip_end_ts']
            faulty_df.loc[neg_indices, 'trip_end_ts'] = temp
        
        # Inject extreme durations
        remaining_idx = faulty_df.index.difference(neg_indices)
        n_extreme = int(len(remaining_idx) * extreme_duration_rate)
        extreme_indices = np.random.choice(remaining_idx, size=n_extreme, replace=False)
        
        if 'duration' in faulty_df.columns:
            faulty_df.loc[extreme_indices, 'duration'] = \
                faulty_df.loc[extreme_indices, 'duration'] * extreme_multiplier
        
        print(f"Injected {n_negative} negative durations and {n_extreme} extreme durations")
        return faulty_df
    else:
        # Spark DataFrame
        faulty_df = df.withColumn(
            "duration",
            F.when(F.rand(seed) < negative_duration_rate, -1 * F.abs(F.col("duration")))
             .when(F.rand(seed + 1) < extreme_duration_rate, F.col("duration") * extreme_multiplier)
             .otherwise(F.col("duration"))
        )
        return faulty_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Case 5: Vehicle ID Foreign Key Issues
# MAGIC **Root Causes:** `silver_vehicle_info_join_miss_rate`, `silver_vehicle_type_nulls`
# MAGIC 
# MAGIC Simulates vehicle master data issues causing join failures.

# COMMAND ----------

def inject_vehicle_id_corruption(
    df: Union[pd.DataFrame, SparkDataFrame],
    corruption_rate: float = 0.30,
    invalid_prefix: str = "INVALID_",
    seed: int = 42
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Corrupt vehicle_id values to cause join failures with vehicle master data.
    
    Args:
        df: Input DataFrame with trips data
        corruption_rate: Fraction of rows to corrupt vehicle_id
        invalid_prefix: Prefix to add to make vehicle_id invalid
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with corrupted vehicle_id values
        
    Expected downstream effects:
        - silver_vehicle_info_join_miss_rate: Increases significantly
        - silver_vehicle_type_nulls: Increases (no vehicle type from join)
    """
    if isinstance(df, pd.DataFrame):
        np.random.seed(seed)
        faulty_df = df.copy()
        
        n_corrupt = int(len(faulty_df) * corruption_rate)
        corrupt_indices = np.random.choice(faulty_df.index, size=n_corrupt, replace=False)
        
        # Corrupt vehicle_id by adding invalid prefix
        if 'vehicle_id' in faulty_df.columns:
            faulty_df.loc[corrupt_indices, 'vehicle_id'] = \
                invalid_prefix + faulty_df.loc[corrupt_indices, 'vehicle_id'].astype(str)
        elif 'unit_id' in faulty_df.columns:
            # Some schemas use unit_id as vehicle identifier
            faulty_df.loc[corrupt_indices, 'unit_id'] = \
                invalid_prefix + faulty_df.loc[corrupt_indices, 'unit_id'].astype(str)
        
        print(f"Corrupted {n_corrupt}/{len(df)} vehicle_id values ({corruption_rate*100:.0f}%)")
        return faulty_df
    else:
        # Spark DataFrame
        faulty_df = df.withColumn(
            "vehicle_id",
            F.when(F.rand(seed) < corruption_rate, 
                   F.concat(F.lit(invalid_prefix), F.col("vehicle_id").cast("string")))
             .otherwise(F.col("vehicle_id"))
        )
        return faulty_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Case 6: Fuel Consumption Anomalies (ML Input)
# MAGIC **Root Causes:** `silver_ml_large_error_count`, `silver_ml_imputed_fuel_p95`
# MAGIC 
# MAGIC Simulates fuel sensor issues that will cause ML model prediction errors.

# COMMAND ----------

def inject_fuel_anomalies(
    df: Union[pd.DataFrame, SparkDataFrame],
    anomaly_rate: float = 0.20,
    null_rate: float = 0.10,
    anomaly_multiplier: float = 5.0,
    seed: int = 42
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Inject fuel consumption anomalies that will trigger ML model errors.
    
    Args:
        df: Input DataFrame with trips data
        anomaly_rate: Fraction of rows with anomalous fuel values
        null_rate: Fraction of rows with NULL fuel values
        anomaly_multiplier: Multiply fuel values for anomalous rows
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with corrupted fuel consumption values
        
    Expected downstream effects:
        - silver_ml_large_error_count: Increases (ML predictions deviate from actuals)
        - silver_ml_imputed_fuel_p95: Increases (high imputed values)
        - p95_fuel_per_100km: Increases
    """
    fuel_cols = ['fuel_consumed', 'fuel_consumption', 'fuel_liters', 'fuel']
    
    if isinstance(df, pd.DataFrame):
        np.random.seed(seed)
        faulty_df = df.copy()
        
        # Find the fuel column
        fuel_col = None
        for col in fuel_cols:
            if col in faulty_df.columns:
                fuel_col = col
                break
        
        if fuel_col is None:
            print("Warning: No fuel column found in DataFrame")
            return faulty_df
        
        # Inject NULLs
        n_nulls = int(len(faulty_df) * null_rate)
        null_indices = np.random.choice(faulty_df.index, size=n_nulls, replace=False)
        faulty_df.loc[null_indices, fuel_col] = None
        
        # Inject anomalous high values
        remaining_idx = faulty_df.index.difference(null_indices)
        n_anomalies = int(len(remaining_idx) * anomaly_rate)
        anomaly_indices = np.random.choice(remaining_idx, size=n_anomalies, replace=False)
        faulty_df.loc[anomaly_indices, fuel_col] = \
            faulty_df.loc[anomaly_indices, fuel_col] * anomaly_multiplier
        
        print(f"Injected {n_nulls} NULL and {n_anomalies} anomalous fuel values")
        return faulty_df
    else:
        # Spark DataFrame - try each possible fuel column
        for fuel_col in fuel_cols:
            if fuel_col in df.columns:
                faulty_df = df.withColumn(
                    fuel_col,
                    F.when(F.rand(seed) < null_rate, F.lit(None))
                     .when(F.rand(seed + 1) < anomaly_rate, F.col(fuel_col) * anomaly_multiplier)
                     .otherwise(F.col(fuel_col))
                )
                return faulty_df
        return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Case 7: Compound Fault - Multiple Raw Issues
# MAGIC **Root Causes:** `raw_null_count_distance`, `raw_null_count_avg_speed`, `raw_null_count_idle_time`
# MAGIC 
# MAGIC Simulates correlated sensor failures affecting multiple columns.

# COMMAND ----------

def inject_compound_sensor_failures(
    df: Union[pd.DataFrame, SparkDataFrame],
    failure_rate: float = 0.30,
    seed: int = 42
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Inject correlated NULL values across multiple sensor columns.
    Simulates a scenario where sensor bus failure affects multiple readings.
    
    Args:
        df: Input DataFrame with trips data
        failure_rate: Fraction of rows affected by sensor failure
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with multiple correlated NULL injections
        
    Expected downstream effects:
        - raw_null_count_distance: Increases
        - raw_null_count_avg_speed: Increases  
        - raw_null_count_idle_time: Increases
        - Multiple bronze/silver metrics affected
    """
    sensor_cols = ['distance', 'avg_speed', 'idle_time', 'fuel_consumed']
    
    if isinstance(df, pd.DataFrame):
        np.random.seed(seed)
        faulty_df = df.copy()
        
        # Select rows affected by "sensor bus failure"
        n_failures = int(len(faulty_df) * failure_rate)
        failure_indices = np.random.choice(faulty_df.index, size=n_failures, replace=False)
        
        # Set all sensor columns to NULL for affected rows
        affected_cols = []
        for col in sensor_cols:
            if col in faulty_df.columns:
                faulty_df.loc[failure_indices, col] = None
                affected_cols.append(col)
        
        print(f"Injected compound sensor failure: {n_failures} rows, columns: {affected_cols}")
        return faulty_df
    else:
        # Spark DataFrame
        # Create a failure flag column
        faulty_df = df.withColumn("_sensor_failure", F.rand(seed) < failure_rate)
        
        for col in sensor_cols:
            if col in df.columns:
                faulty_df = faulty_df.withColumn(
                    col,
                    F.when(F.col("_sensor_failure"), F.lit(None)).otherwise(F.col(col))
                )
        
        faulty_df = faulty_df.drop("_sensor_failure")
        return faulty_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Case 8: Timestamp Anomalies
# MAGIC **Root Causes:** `raw_max_trip_end_ts`, `raw_min_trip_start_ts`
# MAGIC 
# MAGIC Simulates clock synchronization issues.

# COMMAND ----------

def inject_timestamp_anomalies(
    df: Union[pd.DataFrame, SparkDataFrame],
    future_date_rate: float = 0.10,
    past_date_rate: float = 0.10,
    days_offset: int = 365,
    seed: int = 42
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Inject timestamp anomalies (future dates, far past dates).
    
    Args:
        df: Input DataFrame with trips data
        future_date_rate: Fraction of rows with future timestamps
        past_date_rate: Fraction of rows with far past timestamps
        days_offset: Number of days to offset timestamps
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with corrupted timestamps
        
    Expected downstream effects:
        - raw_max_trip_end_ts: May show future dates
        - raw_min_trip_start_ts: May show far past dates
        - bronze validation may drop these rows
    """
    ts_cols = ['trip_start_ts', 'trip_end_ts', 'start_time', 'end_time']
    
    if isinstance(df, pd.DataFrame):
        np.random.seed(seed)
        faulty_df = df.copy()
        
        # Find timestamp columns
        ts_col = None
        for col in ts_cols:
            if col in faulty_df.columns:
                ts_col = col
                break
        
        if ts_col is None:
            print("Warning: No timestamp column found")
            return faulty_df
        
        # Inject future dates
        n_future = int(len(faulty_df) * future_date_rate)
        future_indices = np.random.choice(faulty_df.index, size=n_future, replace=False)
        
        if pd.api.types.is_datetime64_any_dtype(faulty_df[ts_col]):
            faulty_df.loc[future_indices, ts_col] = \
                faulty_df.loc[future_indices, ts_col] + pd.Timedelta(days=days_offset)
        
        # Inject far past dates
        remaining_idx = faulty_df.index.difference(future_indices)
        n_past = int(len(remaining_idx) * past_date_rate)
        past_indices = np.random.choice(remaining_idx, size=n_past, replace=False)
        
        if pd.api.types.is_datetime64_any_dtype(faulty_df[ts_col]):
            faulty_df.loc[past_indices, ts_col] = \
                faulty_df.loc[past_indices, ts_col] - pd.Timedelta(days=days_offset)
        
        print(f"Injected {n_future} future and {n_past} past timestamp anomalies")
        return faulty_df
    else:
        # Spark DataFrame
        from pyspark.sql.functions import date_add, date_sub
        
        for ts_col in ts_cols:
            if ts_col in df.columns:
                faulty_df = df.withColumn(
                    ts_col,
                    F.when(F.rand(seed) < future_date_rate, 
                           F.date_add(F.col(ts_col), days_offset))
                     .when(F.rand(seed + 1) < past_date_rate,
                           F.date_sub(F.col(ts_col), days_offset))
                     .otherwise(F.col(ts_col))
                )
                return faulty_df
        return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fault Injection Registry
# MAGIC 
# MAGIC Master registry mapping test cases to injection functions and parameters.

# COMMAND ----------

FAULT_INJECTION_REGISTRY = {
    "case1_unit_id_nulls": {
        "function": inject_unit_id_nulls,
        "params": {"null_rate": 0.4},
        "description": "40% of unit_id values set to NULL",
        "true_roots": {"raw_null_count_unit_id", "raw_unique_units"},
    },
    "case2_distance_gps_nulls": {
        "function": inject_distance_gps_nulls,
        "params": {"distance_null_rate": 0.35},
        "description": "GPS failures causing NULL distance values",
        "true_roots": {"raw_null_count_distance", "raw_poor_gps_coverage_count"},
    },
    "case3_invalid_speeds": {
        "function": inject_invalid_speeds,
        "params": {"invalid_rate": 0.25, "null_rate": 0.15, "speed_multiplier": 3.0},
        "description": "Invalid avg_speed values (sensor malfunction)",
        "true_roots": {"raw_avg_speed_mean", "raw_null_count_avg_speed"},
    },
    "case4_duration_anomalies": {
        "function": inject_duration_anomalies,
        "params": {"negative_duration_rate": 0.10, "extreme_duration_rate": 0.15},
        "description": "Duration anomalies triggering validation drops",
        "true_roots": {"bronze_rows_dropped_by_rules", "bronze_survival_rate"},
    },
    "case5_vehicle_join_failures": {
        "function": inject_vehicle_id_corruption,
        "params": {"corruption_rate": 0.30},
        "description": "Vehicle ID corruption causing join failures",
        "true_roots": {"silver_vehicle_info_join_miss_rate", "silver_vehicle_type_nulls"},
    },
    "case6_fuel_anomalies": {
        "function": inject_fuel_anomalies,
        "params": {"anomaly_rate": 0.20, "null_rate": 0.10, "anomaly_multiplier": 5.0},
        "description": "Fuel sensor anomalies affecting ML model",
        "true_roots": {"silver_ml_large_error_count", "silver_ml_imputed_fuel_p95"},
    },
    "case7_compound_sensor_failures": {
        "function": inject_compound_sensor_failures,
        "params": {"failure_rate": 0.30},
        "description": "Compound sensor bus failure affecting multiple columns",
        "true_roots": {"raw_null_count_distance", "raw_null_count_avg_speed", "raw_null_count_idle_time"},
    },
    "case8_timestamp_anomalies": {
        "function": inject_timestamp_anomalies,
        "params": {"future_date_rate": 0.10, "past_date_rate": 0.10},
        "description": "Clock sync issues causing timestamp anomalies",
        "true_roots": {"raw_max_trip_end_ts", "raw_min_trip_start_ts"},
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper: Apply Fault Injection

# COMMAND ----------

def apply_fault_injection(
    df: Union[pd.DataFrame, SparkDataFrame],
    case_name: str,
    seed: int = 42
) -> Union[pd.DataFrame, SparkDataFrame]:
    """
    Apply a specific fault injection case to a DataFrame.
    
    Args:
        df: Input DataFrame with clean trips data
        case_name: Name of the test case from FAULT_INJECTION_REGISTRY
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with injected faults
        
    Example:
        faulty_df = apply_fault_injection(clean_df, "case1_unit_id_nulls")
    """
    if case_name not in FAULT_INJECTION_REGISTRY:
        raise ValueError(f"Unknown case: {case_name}. Available: {list(FAULT_INJECTION_REGISTRY.keys())}")
    
    config = FAULT_INJECTION_REGISTRY[case_name]
    inject_fn = config["function"]
    params = config["params"].copy()
    params["seed"] = seed
    
    print(f"\n{'='*60}")
    print(f"APPLYING FAULT: {case_name}")
    print(f"Description: {config['description']}")
    print(f"Expected root causes: {config['true_roots']}")
    print(f"{'='*60}")
    
    faulty_df = inject_fn(df, **params)
    
    return faulty_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Usage

# COMMAND ----------

# Example: How to use for a single test case
# 
# # Load clean data for one day
# clean_df = spark.table("your_raw_trips_table").filter(F.col("date") == "2026-01-15")
# 
# # Apply fault injection
# faulty_df = apply_fault_injection(clean_df.toPandas(), "case1_unit_id_nulls")
# 
# # Convert back and run through ETL
# faulty_spark_df = spark.createDataFrame(faulty_df)
# # ... pass to ETL pipeline ...

print("Fault Injection Logic loaded successfully!")
print(f"\nAvailable test cases: {list(FAULT_INJECTION_REGISTRY.keys())}")
