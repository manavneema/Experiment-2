# Databricks notebook source
import requests
import json
from datetime import datetime
from pyspark.sql import functions as F, types as T, Window, Row, DataFrame
from pyspark.sql.functions import monotonically_increasing_id, col, isnan, count
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PythonModel

# COMMAND ----------

# dbutils.widgets.text("date", defaultValue="2025-12-10", label="Date")
# dbutils.widgets.text("table_name", defaultValue="bms_ds_bronze.trips", label="Table")
# source_table = dbutils.widgets.get("table_name") #bms_ds_prod.bms_ds_dasc.temp_telematics_bronze
run_date = dbutils.widgets.get("date")
print(run_date) 

# COMMAND ----------

#Starting ingestion timer
ingestion_start_ts = datetime.utcnow()

# COMMAND ----------

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

# DBTITLE 1,Untitled
def add_vehicle_info(df: DataFrame, vehicle_info_df: DataFrame) -> DataFrame:
    """
    Add vehicle type and fuel_subtype information to the DataFrame.
    Print row counts before and after join, and completeness stats for vehicle_type and vehicle_fuel_subtype.
    """
    df = df.join(vehicle_info_df, how="left", on=["unit_id"])

    # Completeness metrics (single aggregation)
    metrics = (
        df.agg(
            F.sum(F.col("vehicle_type").isNull().cast("int")).alias("silver_null_vehicle_type_rows"),
            F.sum(F.col("vehicle_fuel_subtype").isNull().cast("int")).alias("silver_null_vehicle_fuel_subtype_rows"),
            F.count("*").alias("silver_total_after_join")
        )
        .collect()[0]
        .asDict()
    )
    
    # Calculate vehicle info join miss rate
    if metrics["silver_total_after_join"] > 0:
        metrics["silver_vehicle_info_join_miss_rate"] = (
            metrics["silver_null_vehicle_type_rows"] / metrics["silver_total_after_join"]
        )
    else:
        metrics["silver_vehicle_info_join_miss_rate"] = 0.0

    return df, metrics

# COMMAND ----------

def preprocess_for_ml_model(df: DataFrame) -> tuple[DataFrame, dict]:
    """
    Add temporal features required by the ML model with metrics tracking.
    Combine different fuel consumption columns into a single fuel_consumption column.
    Uses the exact business logic from the silver pipeline cleaning script.
    Preprocess data for ML model input, handling missing values appropriately.
    Based on the original preprocess_trips function.
    Select and cast features for ML model input.
    """
    metrics = {}

    # Extract day of week (Monday, Tuesday, etc.)
    df = df.withColumn("day_of_week", F.date_format(F.col("start"), "EEEE"))
    
    # Extract time of day categories (CORRECTED)
    df = df.withColumn(
        "time_of_day",
        F.when(
            (F.hour(F.col("start")) >= 0) & (F.hour(F.col("start")) < 6),
            "night"
        )
        .when(
            (F.hour(F.col("start")) >= 6) & (F.hour(F.col("start")) < 12),
            "morning"
        )
        .when(
            (F.hour(F.col("start")) >= 12) & (F.hour(F.col("start")) < 18),
            "afternoon"
        )
        .otherwise("night")  # 18-23 hours
    )

    # ----------------------------
    # Fuel source metrics (before business rules)
    # ----------------------------
    fuel_source_metrics = (
        df.agg(
            F.sum(F.col("fuel_consumption_ecol").isNotNull().cast("int")).alias("silver_fuel_ecol_available"),
            F.sum(F.col("fuel_consumption_ecor").isNotNull().cast("int")).alias("silver_fuel_ecor_available"),
            F.sum(F.col("fuel_consumption_fms_high").isNotNull().cast("int")).alias("silver_fuel_fms_high_available"),
            F.sum(F.col("fuel_consumption_fms_low").isNotNull().cast("int")).alias("silver_fuel_fms_low_available"),
        )
        .collect()[0]
        .asDict()
    )
    metrics.update(fuel_source_metrics)
    
    # Apply business rules for fuel consumption column priority (CORRECTED)
    # When there is fuel consumption EcoLink data and fuel consumption data fms high, discard fuel_consumption_ecol
    df = df.withColumn(
        "fuel_consumption_ecol",
        F.when(F.col("fuel_consumption_fms_high").isNotNull(), F.lit(None)).otherwise(
            F.col("fuel_consumption_ecol")
        ),
    )
    
    # When there is fuel consumption EcoLink data and fuel consumption data fms low, discard fuel_consumption_fms_low
    df = df.withColumn(
        "fuel_consumption_fms_low",
        F.when(F.col("fuel_consumption_ecol").isNotNull(), F.lit(None)).otherwise(
            F.col("fuel_consumption_fms_low")
        ),
    )
    
    # Set fuel_consumption column based on ecol, ecor, fms high or low columns
    df = (
        df.withColumn(
            "fuel_consumption",
            F.when(
                F.col("fuel_consumption_ecol").isNotNull(),
                F.col("fuel_consumption_ecol"),
            ).otherwise(F.col("fuel_consumption")),
        )
        .withColumn(
            "fuel_consumption",
            F.when(
                F.col("fuel_consumption_ecor").isNotNull(),
                F.col("fuel_consumption_ecor"),
            ).otherwise(F.col("fuel_consumption")),
        )
        .withColumn(
            "fuel_consumption",
            F.when(
                F.col("fuel_consumption_fms_high").isNotNull(),
                F.col("fuel_consumption_fms_high"),
            ).otherwise(F.col("fuel_consumption")),
        )
        .withColumn(
            "fuel_consumption",
            F.when(
                F.col("fuel_consumption_fms_low").isNotNull(),
                F.col("fuel_consumption_fms_low"),
            ).otherwise(F.col("fuel_consumption")),
        )
    )
    
    df = df.drop("fuel_consumption_fms_low", "fuel_consumption_ecol", "fuel_consumption_ecor", "fuel_consumption_fms_high")
    
    # ----------------------------
    # avg_speed imputation metrics and processing + Cross-source consistency
    # ----------------------------
    avg_speed_imputed = df.filter(
        F.col("avg_speed").isNull() & 
        F.col("distance").isNotNull() & 
        F.col("duration").isNotNull() & 
        (F.col("duration") > 0)
    ).count()
    
    # Calculate fuel source disagreement rate
    fuel_disagreement = df.filter(
        F.col("fuel_consumption_ecol").isNotNull() &
        F.col("fuel_consumption_fms_high").isNotNull() &
        (F.abs(F.col("fuel_consumption_ecol") - F.col("fuel_consumption_fms_high")) > 
         F.col("fuel_consumption_ecol") * 0.1)  # >10% difference
    ).count()
    
    metrics["silver_avg_speed_imputed"] = avg_speed_imputed
    metrics["silver_fuel_source_disagreement_count"] = fuel_disagreement
    
    # Impute avg_speed for null values using distance and duration (CORRECTED with rounding)
    # avg_speed (km/h) = distance (meters) / duration (seconds) * 3.6
    df = df.withColumn(
        "avg_speed",
        F.when(
            F.col("avg_speed").isNull() & 
            F.col("distance").isNotNull() & 
            F.col("duration").isNotNull() & 
            (F.col("duration") > 0),
            F.round((F.col("distance") * 3.6) / F.col("duration"), 2)
        ).otherwise(F.col("avg_speed"))
    )
    
    # Fill null values in avg_speed with 0 
    df = df.fillna(0, subset=["avg_speed"]).withColumn(
        "idle_time", F.col("idle_time").cast(T.IntegerType())
    )

    # Fill null values in idle_time with 0 
    df = df.fillna(0, subset=["idle_time"]).withColumn(
        "idle_time", F.col("idle_time").cast(T.IntegerType())
    )

    # ----------------------------
    # Categorical null handling
    # ----------------------------
    cat_nulls = (
        df.agg(
            F.sum(F.col("vehicle_type").isNull().cast("int")).alias("silver_vehicle_type_nulls"),
            F.sum(F.col("vehicle_fuel_subtype").isNull().cast("int")).alias("silver_fuel_subtype_nulls"),
        )
        .collect()[0]
        .asDict()
    )
    metrics.update(cat_nulls)

    # For categorical features, ensure no nulls (fill with 'unknown' if needed)
    df = df.fillna({
        "vehicle_type": "unknown",
        "vehicle_fuel_subtype": "unknown"
    })
    
    metrics["silver_output_rows"] = df.count()
    
    return df, metrics

# COMMAND ----------

# DBTITLE 1,Cell 9
def predict_fuel_consumption(
        df: DataFrame, 
        impute_fuel_consumption: bool
    ):
    """
    Predict fuel consumption for each row in the input DataFrame using the new model.
    Uses native Spark ML model loading for optimal performance.
    """

    # Set model registry in order to deploy / fetch models to / from unity catalog
    registry_uri = "databricks-uc"
    mlflow.set_registry_uri(registry_uri)

    # Updated features for the new model
    vehicle_fuel_usage_features = [
        "trip_type",
        "duration",
        "avg_speed",
        "distance_km",
        "idle_time",
        "max_speed",
        "gps_coverage",
        "vehicle_type",
        "vehicle_fuel_subtype",
        "day_of_week",
        "time_of_day",
        "fuel_consumption",
    ]

    df = df.select(*vehicle_fuel_usage_features)

    # Fill nulls in numeric features to match training preprocessing
    numeric_features = [
        "trip_type",
        "duration",
        "avg_speed",
        "distance_km",
        "idle_time",
        "max_speed",
        "gps_coverage"
    ]
    df = df.fillna(0, subset=numeric_features)

    # Load the model from Unity Catalog using native Spark ML loading (FAST!)
    model_name = "bms_ds_prod.bms_ds_dasc.temp_fuel_consumption_gbt_model"
    model_uri = f"models:/{model_name}@blahblah"
    
    print(f"Loading model: {model_name}")
    loaded_model = mlflow.spark.load_model(model_uri)

    # Make predictions using native Spark ML transform (no UDF overhead!)
    print(f"Making predictions on {df.count()} records...")
    predictions = loaded_model.transform(df)

    if impute_fuel_consumption:
        print("impute_fuel_consumption is True")
        result_df = predictions.withColumn("fuel_consumption", F.when(F.col("prediction") < 0, None).otherwise(F.col("prediction")))
        result_df = result_df.drop("prediction")

        impute_stats = result_df.select(
            F.avg("fuel_consumption").alias("silver_ml_imputed_fuel_mean"),
            F.stddev("fuel_consumption").alias("silver_ml_imputed_fuel_std"),
            F.expr("percentile_approx(fuel_consumption, 0.95)").alias("silver_ml_imputed_fuel_p95"),
            F.count(F.when(F.col("fuel_consumption") > 0, 1)).alias("silver_ml_imputation_count")
        ).collect()[0].asDict()
        
        # Drop intermediate columns created by the pipeline
        cols_to_drop = [c for c in result_df.columns if c.endswith("_idx") or c.endswith("_ohe") or c == "features" or c == "rawPrediction"]
        result_df = result_df.drop(*cols_to_drop)
        
        return result_df, impute_stats
    
    else:
        print("impute_fuel_consumption is False")

        result_df = predictions.withColumn(
            "residual", 
            F.col("fuel_consumption") - F.col("prediction")
        ).withColumn(
            "absolute_residual", 
            F.abs(F.col("residual"))
        ).withColumn(
            "percentage_error",
            F.when(F.col("fuel_consumption") > 0, 
                   (F.abs(F.col("residual")) / F.col("fuel_consumption")) * 100
            ).otherwise(None)
        )
        
        residual_stats = result_df.select(
            F.avg("residual").alias("silver_ml_residual_mean"),
            F.stddev("residual").alias("silver_ml_residual_std"),
            F.avg("absolute_residual").alias("silver_ml_abs_residual_mean"),
            F.avg("percentage_error").alias("silver_ml_percentage_error_mean"),
            F.expr("percentile_approx(absolute_residual, 0.95)").alias("silver_ml_abs_residual_p95"),
            F.avg("prediction").alias("silver_ml_prediction_mean"),
            F.stddev("prediction").alias("silver_ml_prediction_std"),
            F.sum(F.when(F.col("absolute_residual") > F.col("fuel_consumption") * 0.5, 1).otherwise(0)).alias("silver_ml_large_error_count"),
            F.count("*").alias("silver_ml_residual_analysis_count")
        ).collect()[0].asDict()

        # Drop intermediate columns created by the pipeline
        cols_to_drop = [c for c in result_df.columns if c.endswith("_idx") or c.endswith("_ohe") or c == "features" or c == "rawPrediction"] + ["mean_residual", "std_residual", "mean_abs_residual", "mean_percentage_error", "p95_abs_residual"]

        result_df = result_df.drop(*cols_to_drop)
        
        return result_df, residual_stats

# COMMAND ----------

def compute_idling_kpi(df):
    """
    Computes idling per 100 km and fuel consumption per 100 km KPIs with summary metrics.
    Assumes df contains: date, idle_time (sec), distance_km, fuel_consumption
    
    Returns:
    - combined_kpi_df: DataFrame with columns [date, metric_name, metric_value]
    - combined_metrics: Dictionary with all summary metrics from both KPIs
    """
    
    # Convert distance from meters to kilometers if needed
    df_with_distance_km = df.withColumn(
        "distance_km", 
        F.when(F.col("distance_km").isNull(), F.col("distance") / 1000.0)
        .otherwise(F.col("distance_km"))
    )

    # Aggregate base quantities for idling KPI
    idling_agg_df = (
        df_with_distance_km.groupBy("date")
        .agg(
            F.sum("idle_time").alias("total_idle_time_sec"),
            F.sum("distance_km").alias("total_distance_km"),
            F.count("*").alias("trip_count")
        )
        .filter(F.col("total_distance_km") > 0)
    )

    # Calculate idling KPI
    idling_kpi_df = idling_agg_df.withColumn(
        "idling_per_100km",
        (F.col("total_idle_time_sec") / F.col("total_distance_km")) * 100
    ).select("date", F.col("idling_per_100km").alias("metric_value")) \
     .withColumn("metric_name", F.lit("idling_per_100km"))

    # Calculate idling KPI metrics
    idling_metrics = (
        idling_agg_df.withColumn(
            "idling_per_100km",
            (F.col("total_idle_time_sec") / F.col("total_distance_km")) * 100
        )
        .agg(
            F.avg("idling_per_100km").alias("mean_idling_per_100km"),
            F.expr("percentile_approx(idling_per_100km, 0.50)").alias("p50_idling_per_100km"),
            F.expr("percentile_approx(idling_per_100km, 0.95)").alias("p95_idling_per_100km")
        )
        .collect()[0]
        .asDict()
    )

    # Aggregate base quantities for fuel KPI
    fuel_agg_df = (
        df_with_distance_km.groupBy("date")
        .agg(
            F.sum("fuel_consumption").alias("total_fuel"),
            F.sum("distance_km").alias("total_distance_km"),
            F.count("*").alias("trip_count")
        )
        .filter((F.col("total_distance_km") > 0) & (F.col("total_fuel").isNotNull()))
    )

    # Calculate fuel KPI
    fuel_kpi_df = fuel_agg_df.withColumn(
        "fuel_per_100km",
        (F.col("total_fuel") / F.col("total_distance_km")) * 100
    ).select("date", F.col("fuel_per_100km").alias("metric_value")) \
     .withColumn("metric_name", F.lit("fuel_per_100km"))

    # Calculate fuel KPI metrics  
    fuel_metrics = (
        fuel_agg_df.withColumn(
            "fuel_per_100km", 
            (F.col("total_fuel") / F.col("total_distance_km")) * 100
        )
        .agg(
            F.avg("fuel_per_100km").alias("mean_fuel_per_100km"),
            F.expr("percentile_approx(fuel_per_100km, 0.50)").alias("p50_fuel_per_100km"),
            F.expr("percentile_approx(fuel_per_100km, 0.95)").alias("p95_fuel_per_100km")
        )
        .collect()[0]
        .asDict()
    )
    
    # Combine DataFrames using union
    combined_kpi_df = idling_kpi_df.union(fuel_kpi_df).select("date", "metric_name", "metric_value")
    
    # Combine metrics dictionaries
    combined_metrics = {**idling_metrics, **fuel_metrics}

    return combined_kpi_df, combined_metrics

# COMMAND ----------

# Defining config and performing basic checks
bronze_table_name = "bms_ds_prod.bms_ds_dasc.temp_telematics_bronze"
silver_table_name = "bms_ds_prod.bms_ds_dasc.temp_telematics_silver"
metrics_table = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics" 
vehicle_info_table = "bms_ds_prod.bms_ds_silver.vehicle_info"

stage = "silver"

assert run_date, "Date widget value is required and cannot be empty."

print(f"Reading data from {bronze_table_name} for date {run_date}.")

fuel_usage_model_name = "bms_ds_prod.bms_ds_dasc.bms_ds_trip_fuel_usage_imputer"
fuel_usage_model_version = "Production"

global_metrics_df = {}
metric_chk = 0

#reading raw table for the date of the current run
bronze_df = spark.table(bronze_table_name).filter(F.col("date") == run_date)
bronze_count = bronze_df.count()
print("Bronze input table count:", bronze_count)

vehicle_info_df = spark.table(vehicle_info_table).select("unit_id", "vehicle_type", "vehicle_fuel_subtype")
# Joining with vehicle info table
vi_df, join_metrics = add_vehicle_info(bronze_df, vehicle_info_df)

metric_chk += len(join_metrics) 
global_metrics_df.update(join_metrics)


vih_count = vi_df.count()
print("After vehicle info join count:", vih_count)

# Processing data to make predictions and calculate KPI
features_df, feature_metrics = preprocess_for_ml_model(vi_df)
data_process_count = features_df.count()
print("Post data processing count:", data_process_count)

# Add survival rate calculation
if vih_count > 0:
    silver_survival_rate = data_process_count / vih_count
else:
    silver_survival_rate = 0.0

global_metrics_df.update({
    "silver_survival_rate": silver_survival_rate,
    "silver_count_after_feature_engineering": data_process_count, 
    "silver_count_after_vehicle_info_join": vih_count, 
    "silver_input_data_count": bronze_count
})
metric_chk += 4

metric_chk += len(feature_metrics) 
global_metrics_df.update(feature_metrics)

#calculating KPI
kpi_df, kpi_metrics = compute_idling_kpi(features_df)
global_metrics_df.update(kpi_metrics)
metric_chk += len(kpi_metrics) 


# COMMAND ----------

imputation_ml_df = features_df.filter(F.col("fuel_consumption").isNull())
residual_ml_df = features_df.filter(F.col("fuel_consumption").isNotNull())

print(f"Imputation dataset (null fuel_consumption): {imputation_ml_df.count()} rows")
print(f"Residual analysis dataset (non-null fuel_consumption): {residual_ml_df.count()} rows")

# Apply model to both datasets
print("\n=== APPLYING FUEL USAGE ML MODEL ===")

print("\n1. Processing imputation dataset...")
imputation_with_predictions, imputed_stats = predict_fuel_consumption(imputation_ml_df, impute_fuel_consumption=True)
print(f"Imputation dataset: {imputation_with_predictions.count()} rows with predictions")
metric_chk += len(imputed_stats) 
global_metrics_df.update(imputed_stats)
print(f"✓ Successfully imputed fuel consumption imputed_stats: {imputed_stats}")

print("\n2. Processing residual analysis dataset...")
residual_with_predictions, residual_stats = predict_fuel_consumption(residual_ml_df, impute_fuel_consumption=False)
print(f"Residual dataset: {residual_with_predictions.count()} rows with residuals")

# Validation: check imputation results
metric_chk += len(residual_stats) 
global_metrics_df.update(residual_stats)
print(f"✓ Residual Analysis Summary:{residual_stats}")

# COMMAND ----------

ingestion_end_ts = datetime.utcnow()
ingestion_duration_sec = (ingestion_end_ts - ingestion_start_ts).total_seconds()
metric_chk = metric_chk + 1 
global_metrics_df.update({"silver_ingestion_duration_sec": ingestion_duration_sec})
print(metric_chk, len(global_metrics_df))
mertics_df = metrics_dict_to_df(global_metrics_df, '2025-12-10', "silver")
write_metrics_df(mertics_df)

# COMMAND ----------
