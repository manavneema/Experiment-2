# Ideal Causal Graph - ETL Pipeline Reference

This document defines the **ground truth causal structure** derived from analyzing the actual ETL pipeline code. It serves as the reference for evaluating any causal discovery algorithm's accuracy.

---

## 1. Methodology

The ideal graph was constructed by:
1. **Code tracing**: Reading each notebook to identify metric definitions
2. **Dependency analysis**: Identifying which metrics are computed FROM which inputs
3. **Business logic mapping**: Understanding transformations, filters, and aggregations

---

## 2. Metrics Inventory

### 2.1 RAW Layer (Notebook 1: Ingestion)

| Metric | Source Code Reference | Description |
|--------|----------------------|-------------|
| `raw_input_record_count` | `record_count = raw_trips_df.count()` | Total records ingested |
| `raw_ingestion_duration_sec` | Timer calculation | Processing time |
| `raw_min_trip_start_ts` | `F.min("start")` | Earliest trip timestamp |
| `raw_max_trip_end_ts` | `F.max("end")` | Latest trip timestamp |
| `raw_distance_mean` | `F.mean("distance")` | Distribution metric |
| `raw_distance_std` | `F.stddev("distance")` | Distribution metric |
| `raw_avg_speed_mean` | `F.mean("avg_speed")` | Distribution metric |
| `raw_avg_speed_std` | `F.stddev("avg_speed")` | Distribution metric |
| `raw_fuel_consumption_mean` | `F.mean("fuel_consumption")` | Distribution metric |
| `raw_fuel_consumption_std` | `F.stddev("fuel_consumption")` | Distribution metric |
| `raw_unique_units` | `F.countDistinct("unit_id")` | Cardinality metric |
| `raw_poor_gps_coverage_count` | `F.sum(gps_coverage < 0.8)` | Quality metric |
| `raw_null_count_{col}` | Per-column null counts | DQ metrics (22 columns) |

### 2.2 BRONZE Layer (Notebook 2: Cleaning)

| Metric | Source Code Reference | Computation |
|--------|----------------------|-------------|
| `bronze_input_rows` | `df.count()` at start | = raw_input_record_count |
| `bronze_correction_trips_removed` | `trip_type == 4` filter | Count of removed rows |
| `bronze_null_primary_key_rows` | `unit_id.isNull() \| start.isNull() \| end.isNull()` | **Direct effect of raw_null_count_unit_id** |
| `bronze_duplicate_rows_removed` | `dropDuplicates()` delta | Deduplication count |
| `bronze_start_after_end_rows` | `start >= end` filter | Validation failures |
| `bronze_distance_km_mean` | `F.mean("distance_km")` | Post-derivation distribution |
| `bronze_distance_km_std` | `F.stddev("distance_km")` | Post-derivation distribution |
| `bronze_duration_mean` | `F.mean("duration")` | Post-derivation distribution |
| `bronze_duration_std` | `F.stddev("duration")` | Post-derivation distribution |
| `bronze_impossible_speed_events` | `avg_speed > 200` count | Quality metric |
| `bronze_negative_fuel_events` | `fuel_consumption < 0` count | Quality metric |
| `bronze_zero_distance_fuel_events` | `distance_km==0 & fuel>0` | Quality metric |
| `bronze_invalid_avg_speed_rows` | Invalid speed condition | Quality metric |
| `bronze_excessive_daily_events_units` | `daily_events >= 124` | Volume anomaly |
| `bronze_idle_time_invalid_corrected` | Invalid idle time count | Correction metric |
| `bronze_output_rows` | `df.count()` at end | Rows after all filters |
| `bronze_survival_rate` | `output_rows / input_rows` | **Key health indicator** |

### 2.3 SILVER Layer (Notebook 3: Transformation)

| Metric | Source Code Reference | Computation |
|--------|----------------------|-------------|
| `silver_input_data_count` | `bronze_df.count()` | = bronze_output_rows |
| `silver_count_after_vehicle_info_join` | Post-join count | After vehicle info lookup |
| `silver_vehicle_info_join_miss_rate` | `null_vehicle_type / total` | Join completeness |
| `silver_null_vehicle_type_rows` | Null count | Join failures |
| `silver_null_vehicle_fuel_subtype_rows` | Null count | Join failures |
| `silver_avg_speed_imputed` | Imputation count | Where `avg_speed = distance/duration*3.6` |
| `silver_fuel_source_disagreement_count` | `\|ecol - fms_high\| > 10%` | Cross-source QA |
| `silver_fuel_*_available` | Source availability counts | Data lineage |
| `silver_vehicle_type_nulls` | Final null count | Post-processing nulls |
| `silver_fuel_subtype_nulls` | Final null count | Post-processing nulls |
| `silver_output_rows` | Final count | After all transformations |
| `silver_survival_rate` | Calculated | Health indicator |
| `silver_ingestion_duration_sec` | Timer | Processing time |

### 2.4 KPI Layer (Notebook 3: Aggregations)

| Metric | Formula | Dependencies |
|--------|---------|--------------|
| `mean_fuel_per_100km` | `(total_fuel / total_distance_km) * 100` | bronze_distance_km_mean (denominator proxy) |
| `p50_fuel_per_100km` | Percentile | Same |
| `p95_fuel_per_100km` | Percentile | Same |
| `mean_idling_per_100km` | `(total_idle_time / total_distance_km) * 100` | bronze_distance_km_mean (denominator proxy) |
| `p50_idling_per_100km` | Percentile | Same |
| `p95_idling_per_100km` | Percentile | Same |

### 2.5 ML Layer (Notebook 3: Predictions)

| Metric | Dependencies |
|--------|--------------|
| `silver_ml_imputed_fuel_mean` | Feature inputs from silver |
| `silver_ml_imputed_fuel_std` | Feature inputs from silver |
| `silver_ml_imputed_fuel_p95` | Feature inputs from silver |
| `silver_ml_prediction_mean` | Model output statistics |
| `silver_ml_prediction_std` | Model output statistics |
| `silver_ml_percentage_error_mean` | Actual vs predicted |
| `silver_ml_residual_std` | Model residuals |

---

## 3. Ideal Causal Edges

### 3.1 Record Flow (Structural)

These edges represent **deterministic** row count propagation:

```
raw_input_record_count ───────► bronze_input_rows           [IDENTITY]
bronze_input_rows ────────────► bronze_output_rows          [FILTERING]  
bronze_output_rows ───────────► silver_input_data_count     [IDENTITY]
silver_input_data_count ──────► silver_output_rows          [TRANSFORMATION]
```

### 3.2 Survival Rate Dependencies (Functional)

```
bronze_input_rows ────────────► bronze_survival_rate        [DENOMINATOR]
bronze_output_rows ───────────► bronze_survival_rate        [NUMERATOR]
silver_input_data_count ──────► silver_survival_rate        [DENOMINATOR]
silver_output_rows ───────────► silver_survival_rate        [NUMERATOR]
```

### 3.3 Data Quality Propagation (Causal)

**Test Case 1: Null unit_id injection**
```
raw_null_count_unit_id ───────► bronze_null_primary_key_rows    [DIRECT CAUSE]
bronze_null_primary_key_rows ─► bronze_output_rows              [FILTER EFFECT]
bronze_output_rows ───────────► bronze_survival_rate            [DERIVED METRIC]
```

**General null propagation pattern:**
```
raw_null_count_{col} ─────────► bronze filtering/correction metrics
                              ► downstream validation failures
```

### 3.4 Distribution Propagation (Statistical)

```
raw_distance_mean ────────────► bronze_distance_km_mean     [UNIT CONVERSION]
raw_distance_std ─────────────► bronze_distance_km_std      [UNIT CONVERSION]
raw_avg_speed_mean ───────────► bronze_duration_mean        [CORRELATED: speed affects duration filtering]
raw_fuel_consumption_mean ────► bronze_negative_fuel_events [DISTRIBUTION TAIL]
```

### 3.5 Cross-Layer Dependencies (Derived)

```
bronze_distance_km_mean ──────► silver_avg_speed_imputed    [FORMULA: speed = dist/dur]
bronze_duration_mean ─────────► silver_avg_speed_imputed    [FORMULA: speed = dist/dur]
bronze_distance_km_mean ──────► mean_fuel_per_100km         [DENOMINATOR]
bronze_distance_km_mean ──────► mean_idling_per_100km       [DENOMINATOR]
```

### 3.6 Cardinality Effects

```
raw_unique_units ─────────────► silver_vehicle_info_join_miss_rate  [MORE UNITS = MORE POTENTIAL MISSES]
raw_unique_units ─────────────► bronze_excessive_daily_events_units [VOLUME RELATIONSHIP]
```

---

## 4. Complete Edge List (Ground Truth)

### 4.1 Tier 1: Deterministic Edges (Must Exist)

| From | To | Relationship | Strength |
|------|-----|--------------|----------|
| raw_input_record_count | bronze_input_rows | Identity | 1.0 |
| bronze_output_rows | silver_input_data_count | Identity | 1.0 |
| bronze_input_rows | bronze_survival_rate | Functional | 1.0 |
| bronze_output_rows | bronze_survival_rate | Functional | 1.0 |
| raw_null_count_unit_id | bronze_null_primary_key_rows | Direct cause | 1.0 |
| raw_null_count_start | bronze_null_primary_key_rows | Direct cause | 1.0 |
| raw_null_count_end | bronze_null_primary_key_rows | Direct cause | 1.0 |

### 4.2 Tier 2: Strong Causal Edges (Should Exist)

| From | To | Relationship |
|------|-----|--------------|
| bronze_null_primary_key_rows | bronze_output_rows | Filter reduces rows |
| bronze_duplicate_rows_removed | bronze_output_rows | Filter reduces rows |
| bronze_start_after_end_rows | bronze_output_rows | Filter reduces rows |
| bronze_rows_dropped_by_rules | bronze_output_rows | Filter reduces rows |
| bronze_excessive_daily_events_units | bronze_output_rows | Filter reduces rows |
| raw_distance_mean | bronze_distance_km_mean | Unit conversion |
| raw_avg_speed_mean | bronze_duration_mean | Speed filtering affects duration |
| bronze_distance_km_mean | silver_avg_speed_imputed | Imputation formula |
| bronze_duration_mean | silver_avg_speed_imputed | Imputation formula |
| bronze_distance_km_mean | mean_fuel_per_100km | Denominator in KPI |
| bronze_distance_km_mean | mean_idling_per_100km | Denominator in KPI |

### 4.3 Tier 3: Statistical Correlation Edges (May Exist)

| From | To | Relationship |
|------|-----|--------------|
| raw_fuel_consumption_mean | bronze_negative_fuel_events | Distribution tail |
| raw_unique_units | silver_vehicle_info_join_miss_rate | Volume effect |
| bronze_impossible_speed_events | silver_avg_speed_imputed | More imputation needed |
| silver_ml_imputed_fuel_mean | silver_ml_prediction_mean | Model I/O |
| silver_ml_imputed_fuel_std | silver_ml_prediction_std | Model I/O |

---

## 5. Visual Representation

```
                              IDEAL ETL CAUSAL GRAPH
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║   RAW LAYER                                                               ║
    ║   ─────────                                                               ║
    ║   ┌─────────────────────┐    ┌─────────────────────┐                      ║
    ║   │raw_input_record_cnt │    │ raw_null_count_*    │ (DQ metrics)         ║
    ║   └──────────┬──────────┘    └──────────┬──────────┘                      ║
    ║              │                          │                                 ║
    ║   ┌──────────┴──────────┐    ┌──────────┴──────────┐                      ║
    ║   │ raw_distance_mean   │    │raw_null_count_unit_id│───────┐             ║
    ║   │ raw_avg_speed_mean  │    │raw_null_count_start │───────┤             ║
    ║   │ raw_unique_units    │    │raw_null_count_end   │───────┤             ║
    ║   └──────────┬──────────┘    └─────────────────────┘       │             ║
    ║              │                                              │             ║
    ║ ═════════════╪══════════════════════════════════════════════╪═════════════║
    ║              │                                              │             ║
    ║   BRONZE LAYER                                              ▼             ║
    ║   ────────────                                 ┌────────────────────────┐ ║
    ║   ┌─────────────────────┐                      │bronze_null_primary_key │ ║
    ║   │  bronze_input_rows  │◄── (= raw_count)     │        _rows           │ ║
    ║   └──────────┬──────────┘                      └───────────┬────────────┘ ║
    ║              │                                             │              ║
    ║              ├────────────────────┬────────────────────────┤              ║
    ║              ▼                    ▼                        ▼              ║
    ║   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐ ║
    ║   │bronze_survival_ │   │bronze_output_   │◄──│ (all filtering effects) │ ║
    ║   │     rate        │◄──│     rows        │   └─────────────────────────┘ ║
    ║   └─────────────────┘   └────────┬────────┘                               ║
    ║                                  │                                        ║
    ║   ┌─────────────────────┐        │    ┌─────────────────────┐             ║
    ║   │bronze_distance_km   │        │    │bronze_duration_mean │             ║
    ║   │      _mean          │────────┼────│                     │             ║
    ║   └─────────┬───────────┘        │    └──────────┬──────────┘             ║
    ║             │                    │               │                        ║
    ║ ════════════╪════════════════════╪═══════════════╪════════════════════════║
    ║             │                    │               │                        ║
    ║   SILVER LAYER                   ▼               │                        ║
    ║   ────────────        ┌─────────────────────┐    │                        ║
    ║                       │silver_input_data_cnt│    │                        ║
    ║                       └──────────┬──────────┘    │                        ║
    ║                                  │               │                        ║
    ║             ┌────────────────────┴───────────────┘                        ║
    ║             ▼                                                             ║
    ║   ┌─────────────────────┐                                                 ║
    ║   │silver_avg_speed_    │◄── (= dist/dur * 3.6)                           ║
    ║   │    imputed          │                                                 ║
    ║   └─────────────────────┘                                                 ║
    ║                                                                           ║
    ║   KPI LAYER                                                               ║
    ║   ─────────                                                               ║
    ║   ┌─────────────────────┐    ┌─────────────────────┐                      ║
    ║   │ mean_fuel_per_100km │◄───│bronze_distance_km   │                      ║
    ║   │mean_idling_per_100km│◄───│     _mean           │ (denominator)        ║
    ║   └─────────────────────┘    └─────────────────────┘                      ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 6. Coverage Metrics Definition

To evaluate any causal discovery algorithm against this ground truth:

### 6.1 Edge-Level Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Edge Recall** | `|Discovered ∩ Ideal| / |Ideal|` | What fraction of ideal edges were found? |
| **Edge Precision** | `|Discovered ∩ Ideal| / |Discovered|` | What fraction of discovered edges are correct? |
| **Edge F1** | `2 * P * R / (P + R)` | Harmonic mean |

### 6.2 Node-Level Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Node Coverage** | `|Nodes in Graph| / |All Metrics|` | Are critical nodes even present? |
| **Critical Node Coverage** | `|Critical Nodes in Graph| / |Critical Nodes|` | Are DQ metrics preserved? |

### 6.3 Path-Level Metrics

| Metric | Description |
|--------|-------------|
| **Causal Path Recall** | Can we trace from raw DQ issues to downstream effects? |
| **RCA Reachability** | From a silver anomaly, can we reach the raw root cause? |

---

## 7. Test Case Ground Truth Mapping

### Test Case 1: 40% Null unit_id

**Expected causal path:**
```
raw_null_count_unit_id (↑) ──► bronze_null_primary_key_rows (↑) ──► bronze_output_rows (↓) ──► bronze_survival_rate (↓)
```

**Root cause:** `raw_null_count_unit_id`
**Observable symptoms:** `bronze_survival_rate`, `bronze_output_rows`, downstream silver metrics

### Test Case 2: Distance Corruption

**Expected causal path:**
```
raw_distance_mean (shifted) ──► bronze_distance_km_mean (shifted) ──► mean_fuel_per_100km (anomalous)
                                                                  ──► mean_idling_per_100km (anomalous)
                                                                  ──► silver_avg_speed_imputed (anomalous)
```

### Test Case 3: Timestamp Issues

**Expected causal path:**
```
raw_null_count_start/end (↑) ──► bronze_null_primary_key_rows (↑) ──► ...
raw_min_trip_start_ts (shifted) ──► temporal_coverage anomalies
```

---

## 8. Critical Metrics for RCA

These metrics **MUST** be preserved in any causal graph for RCA to work:

| Metric | Why Critical |
|--------|--------------|
| `raw_null_count_unit_id` | Root cause for PK validation failures |
| `raw_null_count_start` | Root cause for timestamp validation |
| `raw_null_count_end` | Root cause for timestamp validation |
| `bronze_null_primary_key_rows` | Direct effect of null PKs |
| `bronze_survival_rate` | Key health indicator |
| `silver_survival_rate` | Key health indicator |
| `raw_unique_units` | Cardinality affects join miss rates |
| `bronze_output_rows` | Pipeline throughput indicator |

---

## 9. Appendix: Edge Count Summary

| Category | Count | Examples |
|----------|-------|----------|
| **Tier 1 (Deterministic)** | 7 | Identity edges, functional dependencies |
| **Tier 2 (Strong Causal)** | 11 | Filter effects, formula dependencies |
| **Tier 3 (Statistical)** | 5+ | Distribution correlations |
| **Total Ideal Edges** | ~23-25 | Core structural edges |

This document serves as the reference for evaluating NOTEARS, PC, GraphicalLasso, or any other causal discovery algorithm's accuracy on this ETL pipeline.
