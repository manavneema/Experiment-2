# Databricks notebook source
#import statements
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
from pyspark.ml.feature import VectorAssembler, StandardScaler#creating feature vector
import mlflow.spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading table and creating features

# COMMAND ----------

df = spark.table("bms_ds_prod.bms_ds_dasc.temp_telematics_clean")


print(df.count())
display(df.limit(10))
#31349803

# COMMAND ----------

#removing columns with missing fuel_consumption
base_cols = [
    "date",
    "fuel_consumption",
    "distance_km",
    "duration",
    "avg_speed",
    "idle_time",
    "max_speed",
    "gps_coverage",
    "vehicle_fuel_subtype"
]

df_reg = df.select(base_cols).dropna()
print(df_reg.count())

# COMMAND ----------

null_count = df_reg.filter(F.col("fuel_consumption").isNull()).count()
print(f"Null count in fuel_consumption: {null_count}")

# COMMAND ----------

# DBTITLE 1,Cell 4
#one hot encoding
fuel_types = (
    df_reg.select("vehicle_fuel_subtype")
    .distinct()
    .orderBy("vehicle_fuel_subtype")
    .rdd.flatMap(lambda x: x)
    .collect()
)

print(f"Fuel subtypes used for training: {fuel_types}")

for fuel_type in fuel_types:
    col_name = f"fuel_subtype_{fuel_type.replace(' ', '_').replace('-', '_')}"
    df_reg = df_reg.withColumn(
        col_name,
        F.when(F.col("vehicle_fuel_subtype") == fuel_type, 1.0).otherwise(0.0)
    )

# COMMAND ----------

#creating feature vector
numeric_features = [
    "distance_km",
    "duration",
    "avg_speed",
    "idle_time",
    "max_speed",
    "gps_coverage"
]

fuel_ohe_cols = [c for c in df_reg.columns if c.startswith("fuel_subtype_")]
feature_cols = numeric_features + fuel_ohe_cols

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)

df_features = assembler.transform(
    df_reg.select(
        "date",
        "fuel_consumption",
        *feature_cols
    )
)

display(df_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Splitting test/train data

# COMMAND ----------

df_model = df_reg.select(
    "date",
    "fuel_consumption",
    *feature_cols
)

train_df = df_model.filter(F.col("date") <= "2025-12-14")
test_df  = df_model.filter(F.col("date") >  "2025-12-14")

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Model

# COMMAND ----------

train_df = train_df.withColumn(
    "log_fuel_consumption",
    F.log1p("fuel_consumption")
)

lr = LinearRegression(
    featuresCol="features",
    labelCol="log_fuel_consumption",
    maxIter=50
)

pipeline = Pipeline(stages=[assembler, scaler, lr])

model = pipeline.fit(train_df)

# COMMAND ----------

pred_test = model.transform(test_df)

# 🔒 CLIP LOG PREDICTIONS (CRITICAL FIX)
pred_test = pred_test.withColumn(
    "log_prediction_clipped",
    F.least(F.col("prediction"), F.lit(15.0))
)

# Inverse transform
pred_test = pred_test.withColumn(
    "fuel_prediction",
    F.expm1("log_prediction_clipped")
)

# Residuals (PHYSICAL SPACE)
pred_test = pred_test.withColumn(
    "fuel_residual",
    F.col("fuel_consumption") - F.col("fuel_prediction")
)


# COMMAND ----------

daily_stats = pred_test.groupBy("date").agg(
    F.avg("fuel_residual").alias("avg_fuel_residual"),
    F.stddev("fuel_residual").alias("residual_std")
)

display(daily_stats)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation

# COMMAND ----------

pred_test = model.transform(test_df)

evaluator = RegressionEvaluator(
    labelCol="fuel_consumption",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(pred_test)
print(f"Test RMSE: {rmse}")
#13390

# COMMAND ----------

pred_test = pred_test.withColumn(
    "fuel_residual",
    F.col("fuel_consumption") - F.col("prediction")
)

display(
    pred_test.select(
        "date",
        "fuel_consumption",
        "prediction",
        "fuel_residual"
    ).limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploying the model

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# Run Name: fuel_model_training
# Run ID: 68ad2f58c06a4b5ab49af09ce0756398
# Model URI: runs:/68ad2f58c06a4b5ab49af09ce0756398/model
# You can use this run ID to load the model, view metrics, or reference it in other notebooks. The model was also registered in Unity Catalog as temp_fuel_consumption_lr.

# COMMAND ----------

set_uc_model_registry()

# COMMAND ----------

# DBTITLE 1,Untitled
MODEL_NAME = "temp_fuel_consumption_lr"

with mlflow.start_run(run_name="fuel_model_training"):
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("features", feature_cols)
    mlflow.log_param("fuel_types", fuel_types)

    mlflow.spark.log_model(
        spark_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )

print(f"Model registered: {MODEL_NAME}")

# COMMAND ----------

# DBTITLE 1,Get Run ID
# Get the run ID from the most recent MLflow run
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the experiment ID for this notebook
experiment = mlflow.get_experiment_by_name(f"/Users/manav.neema@bridgestone.com/Lab Day Experiments/Model Setup")

if experiment:
    # Get the most recent run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs:
        run_id = runs[0].info.run_id
        run_name = runs[0].info.run_name
        print(f"Run Name: {run_name}")
        print(f"Run ID: {run_id}")
        print(f"\nModel URI: runs:/{run_id}/model")
    else:
        print("No runs found")
else:
    print("Experiment not found")

# COMMAND ----------

model_version = register_model(
    "68ad2f58c06a4b5ab49af09ce0756398",  # Add run id of the model you want to release
    "bms_ds_prod.bms_ds_dasc.temp_fuel_consumption_lr",
    use_default_artifact_path=True,
    move_to_production=True,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the Registered Model in Another Notebook
# MAGIC
# MAGIC The model `bms_ds_prod.bms_ds_dasc.temp_fuel_consumption_lr` is now registered in Unity Catalog.
# MAGIC
# MAGIC You can load and use it in any notebook with the following approaches:
# MAGIC
# MAGIC ### Option 1: Load the latest version
# MAGIC ```python
# MAGIC import mlflow
# MAGIC from pyspark.ml.feature import VectorAssembler
# MAGIC
# MAGIC # Load the latest version of the model
# MAGIC model_name = "bms_ds_prod.bms_ds_dasc.temp_fuel_consumption_lr"
# MAGIC model = mlflow.spark.load_model(f"models:/{model_name}/latest")
# MAGIC
# MAGIC # Or load a specific version
# MAGIC # model = mlflow.spark.load_model(f"models:/{model_name}/1")
# MAGIC
# MAGIC # Or load the production version
# MAGIC # model = mlflow.spark.load_model(f"models:/{model_name}/Production")
# MAGIC ```
# MAGIC
# MAGIC ### Option 2: Prepare features and make predictions
# MAGIC ```python
# MAGIC # Load your data
# MAGIC df = spark.table("bms_ds_prod.bms_ds_dasc.temp_telematics_clean")
# MAGIC
# MAGIC # Create the same features used during training
# MAGIC fuel_types = ['biodiesel_fame', 'commercial_diesel', 'diesel', 'hvo', 'petrol']
# MAGIC
# MAGIC for fuel_type in fuel_types:
# MAGIC     col_name = f"fuel_subtype_{fuel_type}"
# MAGIC     df = df.withColumn(
# MAGIC         col_name,
# MAGIC         F.when(F.col("vehicle_fuel_subtype") == fuel_type, 1.0).otherwise(0.0)
# MAGIC     )
# MAGIC
# MAGIC # Assemble features
# MAGIC numeric_features = ["distance_km", "duration", "avg_speed", "idle_time", "max_speed", "gps_coverage"]
# MAGIC fuel_ohe_cols = [f"fuel_subtype_{ft}" for ft in fuel_types]
# MAGIC feature_cols = numeric_features + fuel_ohe_cols
# MAGIC
# MAGIC assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# MAGIC df_features = assembler.transform(df)
# MAGIC
# MAGIC # Make predictions
# MAGIC predictions = model.transform(df_features)
# MAGIC
# MAGIC display(predictions.select("date", "fuel_consumption", "prediction"))
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # FINAL MODEl SETUP

# COMMAND ----------

# from pyspark.ml import Pipeline
# from pyspark.ml.feature import (
#     StringIndexer,
#     OneHotEncoder,
#     VectorAssembler
# )
# from pyspark.ml.regression import GBTRegressor
# from pyspark.sql import functions as F
# import mlflow
# import mlflow.spark

# numeric_features = [
#     "trip_type",
#     "duration",
#     "avg_speed",
#     "distance_km",
#     "idle_time",
#     "max_speed",
#     "gps_coverage"
# ]

# categorical_features = [
#     "vehicle_type",
#     "vehicle_fuel_subtype",
#     "day_of_week",
#     "time_of_day"
# ]

# indexers = [
#     StringIndexer(
#         inputCol=col,
#         outputCol=f"{col}_idx",
#         handleInvalid="keep"
#     )
#     for col in categorical_features
# ]

# encoders = [
#     OneHotEncoder(
#         inputCol=f"{col}_idx",
#         outputCol=f"{col}_ohe"
#     )
#     for col in categorical_features
# ]

# feature_cols = numeric_features + [f"{c}_ohe" for c in categorical_features]

# assembler = VectorAssembler(
#     inputCols=feature_cols,
#     outputCol="features",
#     handleInvalid="skip"
# )

# gbt = GBTRegressor(
#     labelCol="fuel_consumption",
#     featuresCol="features",
#     maxDepth=5,
#     maxIter=50,
#     stepSize=0.05,
#     seed=42
# )

# pipeline = Pipeline(stages=indexers + encoders + [assembler, gbt])

# residual_ml_df = residual_ml_df.fillna(0, subset=numeric_features)
# train_df, test_df = residual_ml_df.randomSplit([0.7, 0.3], seed=42)
# print(f"Train count: {train_df.count()}")
# print(f"Test count: {test_df.count()}")

# mlflow.set_registry_uri("databricks-uc")

# with mlflow.start_run(run_name="fuel_consumption_gbt") as run:
#     model = pipeline.fit(train_df)

#     mlflow.log_param("model_type", "GBTRegressor")
#     mlflow.log_param("maxDepth", 5)
#     mlflow.log_param("maxIter", 50)
#     mlflow.log_param("features", feature_cols)
    
#     # Make predictions to infer signature
#     predictions = model.transform(train_df.limit(100))
    
#     # Infer signature from input features and predictions
#     from mlflow.models import infer_signature
#     signature = infer_signature(train_df.limit(100), predictions)
    
#     # Log model with signature
#     mlflow.spark.log_model(
#         model,
#         artifact_path="model",
#         signature=signature
#     )
    
#     print(f"Model logged with run_id: {run.info.run_id}")

# # Register the model from the previous run
# mlflow.register_model(
#     model_uri=f"runs:/{run.info.run_id}/model",
#     name="bms_ds_prod.bms_ds_dasc.temp_fuel_consumption_gbt_model"
# )

# # Load the registered model and make predictions
# from pyspark.ml.evaluation import RegressionEvaluator

# # Load the model from Unity Catalog
# model_name = "bms_ds_prod.bms_ds_dasc.temp_fuel_consumption_gbt_model"
# model_uri = f"models:/{model_name}@blahblah"

# print(f"Loading model: {model_name}")
# loaded_model = mlflow.spark.load_model(model_uri)

# # Make predictions on test data
# print(f"\nMaking predictions on {test_df.count()} test records...")
# predictions = loaded_model.transform(test_df)

# # Show sample predictions
# print("\nSample predictions:")
# display(predictions.select(
#     "unit_id",
#     "fuel_consumption",
#     "prediction",
#     "avg_speed",
#     "distance_km",
#     "duration",
#     "vehicle_type"
# ).limit(20))

# # Evaluate model performance
# evaluator_rmse = RegressionEvaluator(
#     labelCol="fuel_consumption",
#     predictionCol="prediction",
#     metricName="rmse"
# )

# evaluator_mae = RegressionEvaluator(
#     labelCol="fuel_consumption",
#     predictionCol="prediction",
#     metricName="mae"
# )

# evaluator_r2 = RegressionEvaluator(
#     labelCol="fuel_consumption",
#     predictionCol="prediction",
#     metricName="r2"
# )

# rmse = evaluator_rmse.evaluate(predictions)
# mae = evaluator_mae.evaluate(predictions)
# r2 = evaluator_r2.evaluate(predictions)

# print("\n=== MODEL PERFORMANCE METRICS ===")
# print(f"RMSE: {rmse:.2f}")
# print(f"MAE: {mae:.2f}")
# print(f"R²: {r2:.4f}")

# # Calculate additional statistics
# prediction_stats = predictions.select(
#     F.mean("fuel_consumption").alias("actual_mean"),
#     F.mean("prediction").alias("predicted_mean"),
#     F.stddev("fuel_consumption").alias("actual_stddev"),
#     F.stddev("prediction").alias("predicted_stddev")
# ).collect()[0]

# print(f"\nActual fuel consumption - Mean: {prediction_stats['actual_mean']:.2f}, StdDev: {prediction_stats['actual_stddev']:.2f}")
# print(f"Predicted fuel consumption - Mean: {prediction_stats['predicted_mean']:.2f}, StdDev: {prediction_stats['predicted_stddev']:.2f}")

# def preprocess_for_ml_model_og(df: DataFrame) -> DataFrame:
#     """
#     Add temporal features required by the ML model.
#     Combine different fuel consumption columns into a single fuel_consumption column.
#     Uses the exact business logic from the silver pipeline cleaning script.
#     Preprocess data for ML model input, handling missing values appropriately.
#     Based on the original preprocess_trips function.
#     Select and cast features for ML model input.
#     """

#     # Extract day of week (Monday, Tuesday, etc.)
#     df = df.withColumn("day_of_week", F.date_format(F.col("start"), "EEEE"))
    
#     # Extract time of day categories
#     df = df.withColumn(
#         "time_of_day",
#         F.when(
#             (F.hour(F.col("start")) >= 0) & (F.hour(F.col("start")) < 6),
#             "night"
#         )
#         .when(
#             (F.hour(F.col("start")) >= 6) & (F.hour(F.col("start")) < 12),
#             "morning"
#         )
#         .when(
#             (F.hour(F.col("start")) >= 12) & (F.hour(F.col("start")) < 18),
#             "afternoon"
#         )
#         .otherwise("night")  # 18-23 hours
#     )

#     # Apply business rules for fuel consumption column priority
#     # When there is fuel consumption EcoLink data and fuel consumption data fms high, discard fuel_consumption_ecol
#     df = df.withColumn(
#         "fuel_consumption_ecol",
#         F.when(F.col("fuel_consumption_fms_high").isNotNull(), F.lit(None)).otherwise(
#             F.col("fuel_consumption_ecol")
#         ),
#     )
    
#     # When there is fuel consumption EcoLink data and fuel consumption data fms low, discard fuel_consumption_fms_low
#     df = df.withColumn(
#         "fuel_consumption_fms_low",
#         F.when(F.col("fuel_consumption_ecol").isNotNull(), F.lit(None)).otherwise(
#             F.col("fuel_consumption_fms_low")
#         ),
#     )
    
#     # Set fuel_consumption column based on ecol, ecor, fms high or low columns
#     df = (
#         df.withColumn(
#             "fuel_consumption",
#             F.when(
#                 F.col("fuel_consumption_ecol").isNotNull(),
#                 F.col("fuel_consumption_ecol"),
#             ).otherwise(F.col("fuel_consumption")),
#         )
#         .withColumn(
#             "fuel_consumption",
#             F.when(
#                 F.col("fuel_consumption_ecor").isNotNull(),
#                 F.col("fuel_consumption_ecor"),
#             ).otherwise(F.col("fuel_consumption")),
#         )
#         .withColumn(
#             "fuel_consumption",
#             F.when(
#                 F.col("fuel_consumption_fms_high").isNotNull(),
#                 F.col("fuel_consumption_fms_high"),
#             ).otherwise(F.col("fuel_consumption")),
#         )
#         .withColumn(
#             "fuel_consumption",
#             F.when(
#                 F.col("fuel_consumption_fms_low").isNotNull(),
#                 F.col("fuel_consumption_fms_low"),
#             ).otherwise(F.col("fuel_consumption")),
#         )
#     )
    
#     df = df.drop("fuel_consumption_fms_low", "fuel_consumption_ecol", "fuel_consumption_ecor", "fuel_consumption_fms_high")
    
#     # Impute avg_speed for null values using distance and duration
#     # avg_speed (km/h) = distance (meters) / duration (seconds) * 3.6
#     df = df.withColumn(
#         "avg_speed",
#         F.when(
#             F.col("avg_speed").isNull() & 
#             F.col("distance").isNotNull() & 
#             F.col("duration").isNotNull() & 
#             (F.col("duration") > 0),
#             F.round((F.col("distance") * 3.6) / F.col("duration"), 2)
#         ).otherwise(F.col("avg_speed"))
#     )
    
#     # Fill null values in avg_speed with 0 
#     df = df.fillna(0, subset=["avg_speed"]).withColumn(
#         "idle_time", F.col("idle_time").cast(T.IntegerType())
#     )

#     # Fill null values in idle_time with 0 
#     df = df.fillna(0, subset=["idle_time"]).withColumn(
#         "idle_time", F.col("idle_time").cast(T.IntegerType())
#     )

#     # For categorical features, ensure no nulls (fill with 'unknown' if needed)
#     df = df.fillna({
#         "vehicle_type": "unknown",
#         "vehicle_fuel_subtype": "unknown"
#     })
    
#     return df

# COMMAND ----------

