# ETL Notebooks - Local Windows Development Setup

Converted Databricks ETL notebooks to pandas-based Jupyter notebooks for local Windows development.

## Overview

This directory contains 3 ETL notebooks that form a data pipeline:

1. **1_trips_data_ingestion_notebook.ipynb** - Load and ingest raw trips data
2. **2_trips_cleaning_notebook.ipynb** - Validate and clean data (bronze layer)
3. **3_trips_transformation_notebook.ipynb** - Transform for ML and compute KPIs (silver layer)

All notebooks have been converted from PySpark to pandas and run locally without Databricks.

## Setup Instructions

### 1. Create Python Environment

**Using venv (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n etl-local python=3.9
conda activate etl-local
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Sample Data

Before running notebooks, generate sample test data:

```bash
python generate_sample_data.py
```

This creates:
- `artifacts/trips_raw.parquet` - Sample raw trips data
- `artifacts/vehicle_info.parquet` - Sample vehicle metadata

### 4. Run Notebooks

**Option A: Using Jupyter**
```bash
jupyter notebook
```
Then open each notebook in sequence:
1. `1_trips_data_ingestion_notebook.ipynb`
2. `2_trips_cleaning_notebook.ipynb`
3. `3_trips_transformation_notebook.ipynb`

**Option B: Using JupyterLab**
```bash
jupyter lab
```

## Pipeline Architecture

### Data Layers

```
Raw Data (trips_raw.parquet)
         ↓
    [Notebook 1: Ingestion]
         ↓
Raw Table (trips_raw.parquet - metrics tracked)
         ↓
    [Notebook 2: Cleaning]
         ↓
Bronze Table (trips_bronze.parquet)
         ↓
    [Notebook 3: Transformation]
         ↓
Silver Table (trips_silver.parquet)
```

### Output Files

All notebooks save outputs to `./artifacts/`:

- `trips_raw.parquet` - Raw ingested data with metrics
- `trips_bronze.parquet` - Cleaned data (bronze layer)
- `trips_silver.parquet` - Transformed data with ML predictions (silver layer)
- `vehicle_info.parquet` - Vehicle metadata
- `metrics.parquet` - Aggregated metrics from all pipeline stages

## Configuration

Each notebook has a Configuration cell at the top:

```python
ARTIFACT_PATH = "./artifacts"          # Output directory
RUN_DATE = "2025-12-15"               # Date to process
MIN_DURATION_LIMIT = 31               # Minimum trip duration (seconds)
MAX_DAILY_EVENTS = 124                # Max trips per vehicle per day
MAX_DURATION = 86400.0                # Max trip duration (seconds)
MAX_DISTANCE_KM = 1200.0              # Max trip distance (km)
```

Modify these values before running notebooks as needed.

## Notebook Details

### Notebook 1: Data Ingestion
- Loads raw trips data from parquet
- Filters by run_date
- Computes distribution metrics (mean, std, count, unique units)
- Tracks null counts and GPS coverage
- Outputs: `trips_raw.parquet`, metrics appended to `metrics.parquet`

### Notebook 2: Data Cleaning
- Loads raw data from notebook 1
- Validates and filters trips:
  - Removes correction trips (trip_type = 4)
  - Drops null primary keys
  - Removes duplicates
  - Validates start < end timestamps
  - Filters by duration and distance rules
  - Validates average speed
  - Removes excessive daily events
  - Corrects invalid idle times
- Outputs: `trips_bronze.parquet`, metrics appended to `metrics.parquet`

### Notebook 3: Data Transformation
- Loads bronze data from notebook 2
- Joins vehicle information
- Preprocesses features for ML:
  - Derives distance_km, duration, avg_speed
  - Fills missing engine_cc with median
- Predicts fuel consumption using ML model
  - Uses scikit-learn RandomForest as placeholder
  - In production: would load from MLflow
- Computes KPIs:
  - Idle percentage
  - Fuel efficiency (L/100km)
  - Residual analysis
- Outputs: `trips_silver.parquet`, metrics appended to `metrics.parquet`

## Key Conversions (PySpark → Pandas)

| PySpark | Pandas |
|---------|--------|
| `spark.sql(query)` | `pd.read_parquet()` / `pd.read_csv()` |
| `spark.table('name')` | `pd.read_parquet(path)` |
| `F.col('col')` | `df['col']` or `df.loc[]` |
| `F.when().otherwise()` | `np.where()` or `df.where()` |
| `Window.partitionBy()` | `groupby().transform()` |
| `.filter()` | `.loc[]` or `.query()` |
| `.withColumn()` | `.assign()` or direct assignment |
| `.dropDuplicates()` | `.drop_duplicates()` |
| `spark.write.mode('append')` | `pd.concat()` + `.to_parquet()` |
| `dbutils.widgets.get()` | Function parameters / hardcoded config |

## Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'pandas'"**
- Solution: Run `pip install -r requirements.txt`

**Issue: "FileNotFoundError: artifacts/trips_raw.parquet"**
- Solution: Run `python generate_sample_data.py` to create sample data

**Issue: Notebook kernel not found**
- Solution: Install jupyter: `pip install jupyter ipython`
- Then restart notebook and select Python 3 kernel

**Issue: Memory errors with large datasets**
- Solution: Modify sample data generator to use fewer rows:
  ```bash
  python -c "from generate_sample_data import generate_sample_data; generate_sample_data(num_trips=100, num_vehicles=10)"
  ```

## Development Notes

- All notebooks are fully self-contained (no dependencies on Databricks)
- Each notebook can be run independently (given required input files exist)
- ML model in notebook 3 uses scikit-learn RandomForest as placeholder
  - Replace with actual model path as needed
  - In production: would use MLflow server
- Sample data generator uses fixed random seed (42) for reproducibility
- All timestamps are in UTC (datetime.utcnow())

## Next Steps

1. **Replace sample data** - Use real data path in configuration
2. **Update RUN_DATE** - Change to your target processing date
3. **Tune thresholds** - Adjust MIN_DURATION_LIMIT, MAX_DAILY_EVENTS, etc. for your use case
4. **Production ML model** - Replace placeholder RandomForest with actual trained model
5. **Monitor metrics** - Check `metrics.parquet` for data quality insights

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review notebook markdown cells for step-by-step documentation
3. Check output of each cell for error messages
4. Verify all required parquet files exist in `./artifacts/`

---

**Last Updated:** December 2024
**Python Version:** 3.9+
**Framework:** Pandas (no PySpark, no Databricks required)
