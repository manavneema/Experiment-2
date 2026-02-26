# Graph Analysis: Critical Issues Affecting RCA Performance

**Date**: February 26, 2026  
**Current Performance**: 54.5% Top-5 Accuracy (12/22 test cases)  
**Analysis Scope**: Graph v3 artifacts + discovery notebook

---

## Executive Summary

Your causal graph has **6 critical structural issues** that explain why RCA performance plateaus at 54.5%. The problems are not about algorithm bugs, but about **what the graph is actually representing vs. what root causes need**.

| Issue | Impact | Severity | Fix Complexity |
|-------|--------|----------|-----------------|
| Hub node dominance is LEGITIMATE, not spurious | Timestamps + distance always ranked first | HIGH | Medium |
| **3 failure patterns missing from graph entirely** | 7/22 test cases fail (fuel, predictions, duplicates) | CRITICAL | High |
| Graph prefers downstream propagation over direct causation | Misses anomalies not flowing through hubs | HIGH | High |
| Negative weight edges confuse score accumulation | May suppress correct candidates | MEDIUM | Medium |
| Tier assignment mismatch (nodes exist in tiers but not edges) | Config doesn't match final graph | MEDIUM | Low |
| Pattern priors inflate hub importance | 23 priors added, mostly from 2 nodes | MEDIUM | Medium |

---

## Issue 1: Hub Node Dominance is LEGITIMATE (Not Spurious)

### What's Happening

Your top-ranked nodes are **source nodes with no incoming edges**:

```
bronze_distance_km_mean:    12 outgoing edges, 0 incoming ✓ SOURCE
bronze_duration_mean:        9 outgoing edges, 0 incoming ✓ SOURCE
raw_max_trip_end_ts:         2 outgoing edges, 1 incoming (semi-source)
```

These dominate scores because **they reach 12+ downstream anomalies**:
- Distance flows to: duration_std, fuel metrics, speed metrics, GPS metrics, etc.
- Duration flows to: rows_dropped, ingestion_duration, speed metrics, etc.

### Why This Is Correct (Not a Bug)

In a real data pipeline:
1. **Distance traveled** determines expected fuel consumption, speed, idling
2. **Trip duration** determines ingestion latency, validation rules, data quality checks
3. **Timestamps** control when data enters the system

**These are genuine causal ancestors.** Any anomaly in distance/duration ripples downstream.

### The Real Problem

Your graph **assumes all anomalies flow downstream through these hubs**, but real faults don't work that way:

```
Test Case: case3_fuel_sensor_drift (Fails at rank 37)
  True root: p95_fuel_per_100km
  Problem: Fuel sensor drift doesn't depend on trip distance
           It's a sensor malfunction (external to the distance→fuel causal path)
           
Test Case: case5_sensor_reads_nulls (Fails at rank 38)
  True root: raw_mean_fuel_consumption_ecol
  Problem: Data corruption at the source isn't caused by distance
           It's a data ingestion failure

Test Case: case13_aggregation_errors (Fails at rank 27)
  True root: bronze_duplicate_rows_removed
  Problem: Duplicate rows aren't caused by distance/duration
           They're caused by ETL logic errors
```

### Evidence of the Problem

**Successful cases (hub nodes ARE the root)**:
- case12_timestamp_inconsistencies: ✓ Rank 1 (raw_max_trip_end_ts IS a timestamp issue)
- case4_clock_skew: ✓ Rank 1 (raw_max_trip_end_ts IS a clock issue)
- case8_invalid_ranges: ✓ Rank 4 (bronze_duration_mean IS a range validation issue)

**Failed cases (root is independent of hubs)**:
- case3, case5: Fuel metrics (independent sensor)
- case9, case10, case15, case19, case20: Prediction metrics (ML model, not pipeline)
- case13, case14: Data quality (ETL logic, not distance/duration)

---

## Issue 2: CRITICAL - Missing Nodes (7 of 22 Test Cases Fail)

### What's Missing From Final Graph

Three categories of metrics exist in your data but **are not in the final 55-edge graph**:

#### A. Fuel Metrics (3 test cases fail)

**Missing edges for**:
- `raw_std_fuel_consumption_ecol` (test cases 3, 7, 13, 17)
- `raw_mean_fuel_consumption_ecol` (test case 5)
- `p95_fuel_per_100km` (test case 3, 10, 16, 20) ← **Even tier assignments mention this!**

**Why it matters**:
- Fuel sensor failures are **independent root causes**
- They don't flow from distance/duration
- Your graph assumes fuel metrics are computed FROM distance
- **Reality**: Fuel metrics are sensor readings that can fail independently

**Graph evidence**:
```python
# In causal_artifacts-2.json:
"tier_assignments": {
  "p95_fuel_per_100km": 3,  ← Listed in tiers
  "raw_std_fuel_consumption_ecol": 0  ← Tier 0 (raw data)
}

# In hybrid_causal_edges-2.csv:
# NO EDGES ending in these fuel metrics
# bronze_distance_km_mean → p95_fuel_per_100km exists (weight -0.131)
# But this is ONE edge, and it's NEGATIVE (fuel decreases with distance)
```

#### B. Prediction Model Metrics (4 test cases fail)

**Missing proper connections for**:
- `silver_ml_prediction_std` (test case 5, 15)
- `silver_ml_large_error_count` (test case 5, 10, 15, 16, 20)

**Why it matters**:
- ML prediction errors are **independent of distance/duration**
- They depend on: model quality, training data distribution, input feature outliers
- Your graph connects them only through `bronze_excessive_daily_events_units`

**Graph evidence**:
```python
# Path to reach silver_ml_large_error_count:
bronze_excessive_daily_events_units → silver_ml_large_error_count
# That's THE ONLY source, via bootstrap_stable edge
# But that's a weak signal for model errors

# Missing: direct connection from distance/duration anomalies
# Would need: raw data quality → feature computation → model input → prediction error
```

#### C. Data Quality Metrics (2 test cases fail)

**Missing edges for**:
- `bronze_duplicate_rows_removed` (test cases 13, 15)
- No direct connection to duplicate detection

**Why it matters**:
- Duplicates are **ETL-specific**, not distance/duration dependent
- They result from merge failures, not measurement anomalies

---

## Issue 3: Graph Assumes Downstream Propagation (Wrong Causality Model)

### The Problem

Your graph assumes:
```
Raw Metrics → Bronze Metrics → Silver/KPI Metrics
     ↓            ↓                 ↓
  (source)   (aggregated)      (computed)
```

**This works for**: Metrics that flow through the pipeline sequentially  
**This breaks for**: Anomalies that occur at the SOURCE and DON'T propagate

### Evidence: Test Case Failures

**Scenario 1: Source Anomaly That Doesn't Propagate**
```
Test case: case1_unit_id_nulls
True root: raw_null_count_unit_id (Rank 3, DETECTED)
Why it's detected:
  - raw_null_count_unit_id IS in graph
  - Downstream: bronze_ingestion_duration_sec, bronze_null_primary_key_rows
  - These downstream nodes have anomalies too
  ✓ Cascade effect allows detection

Test case: case5_sensor_reads_nulls  
True root: raw_mean_fuel_consumption_ecol (Rank 38, MISSED)
Why it's NOT detected:
  - raw_mean_fuel_consumption_ecol is NOT in graph at all
  - No downstream path exists
  - Even if added, fuel doesn't cascade like null_counts
  ✗ No propagation path means rank 38
```

**Scenario 2: Computed Anomaly (Backward Causality Needed)**
```
Test case: case10_prediction_outliers
True root: silver_ml_large_error_count (Rank 13, MISSED)
Why it's NOT detected:
  - silver_ml_large_error_count IS in graph
  - But it has ONLY 1 incoming edge (from bronze_excessive_daily_events_units)
  - RCA traversal reaches it via: bronze_distance_km_mean → ... → bronze_excessive_daily_events_units
  - This is VERY indirect (5+ hops)
  ✗ By the time RCA reaches it, many other candidates are ranked higher
```

### Root Cause

Your graph structure says:
```
"If distance/duration are anomalous, downstream metrics become anomalous"
```

But RCA root cause says:
```
"Which metric would explain ALL the observed anomalies?"
```

These are **different questions**. Your graph answers the first; RCA needs the second.

---

## Issue 4: Negative Weight Edges Confuse Scoring

### The Problem

Multiple edges have **negative weights**:
```
bronze_distance_km_mean → silver_avg_speed_imputed:  -0.186
bronze_distance_km_mean → raw_null_count_avg_speed:  -0.201
silver_ml_prediction_mean → silver_ml_residual_mean: -0.017
```

**In RCA context, this causes**:
- Negative correlations reduce scores
- If distance↓ correlates with speed↑, a distance anomaly WON'T trigger speed metric ranking
- **You lose anomalies that anti-correlate with hubs**

### Example From Results

```
Test case: case2_distance_gps_nulls
True roots: raw_null_count_gps_coverage (Rank 9), raw_null_count_start (Rank 38)
Status: ✗ MISSED (only ranked 9 in top-5)

Edge: bronze_distance_km_mean → raw_null_count_gps_coverage (negative weight -0.201)
Problem:
  - If distance is high, null_count_gps goes down
  - GPS nulls (actual anomaly) don't propagate through distance metric
  - Negative correlation SUPPRESSES the GPS null metric's score
```

---

## Issue 5: Tier Assignments Don't Match Final Graph

### The Discrepancy

**In causal_artifacts-2.json**:
```json
"tier_assignments": {
  "p95_fuel_per_100km": 3,           ← Tier 3 (KPI)
  "p95_idling_per_100km": 3,         ← Tier 3 (KPI)
  "silver_ml_prediction_std": 3,     ← Tier 3
  "silver_ml_large_error_count": 3   ← Tier 3
}
```

**In final graph (55 edges)**:
- `p95_fuel_per_100km`: HAS 1 incoming edge (distance → fuel)
- `p95_idling_per_100km`: HAS 1 incoming edge (distance → idling)
- `silver_ml_prediction_std`: HAS 2 incoming edges (from ML prediction/error)
- `silver_ml_large_error_count`: HAS 1 incoming edge (from excessive events)

**BUT**: These tier assignments were used to **filter out connections** during discovery.

### Why This Matters

Your tier constraints say:
```python
TIER_JUMP_THRESHOLD = 2  # Forbid edges skipping 2+ tiers
```

But you're assigning:
- Fuel metrics to **Tier 3** (KPI)
- Raw metrics to **Tier 0** (Raw)
- Bronze metrics to **Tier 1**
- Silver metrics to **Tier 2**

**Result**: Raw → KPI jumps 3 tiers, which would be forbidden **if enforced consistently**.

This suggests:
1. Either tier assignments are wrong (fuel shouldn't be Tier 3)
2. Or tier constraints aren't enforced (they're not)
3. Or the graph has edges that violate tier constraints

---

## Issue 6: Pattern Priors Inflate Hub Importance

### What Happened

Your discovery process added **23 pattern-based structural priors**:

```python
# From generate_pattern_based_priors():
# PATTERN 1: raw_X → bronze_X
# PATTERN 2: bronze_X → silver_X
# PATTERN 3: null_count → validation_metrics
# PATTERN 4: distance/duration → computed_metrics  ← THIS ONE inflates hubs
# PATTERN 5: ingestion_duration across tiers
```

**Result: These priors added edges like**:
- `bronze_distance_km_mean → p95_fuel_per_100km`
- `bronze_distance_km_mean → p95_idling_per_100km`
- `bronze_duration_mean → silver_ingestion_duration_sec`
- (... many more)

### Why This Is Problematic

**Pattern 4 (distance/duration → computed metrics) assumes**:
```
"Any metric with 'fuel', 'speed', or 'idling' in the name is computed FROM distance/duration"
```

**But reality is**:
- `p95_fuel_per_100km` is computed FROM fuel sensor readings (raw_std_fuel_consumption_ecol)
- `raw_avg_speed_mean` is computed FROM GPS data (raw data)
- Distance is ONE input, but not the ONLY input

**This inflates** the downstream importance of distance/duration nodes.

### Evidence

From `pattern_prior_edges.csv`:
```
bronze_distance_km_mean,raw_null_count_avg_speed
bronze_distance_km_mean,silver_vehicle_info_join_miss_rate
bronze_distance_km_mean,p95_fuel_per_100km
bronze_distance_km_mean,p95_idling_per_100km
...
bronze_duration_mean,raw_avg_speed_mean
bronze_duration_mean,silver_ingestion_duration_sec
...
(23 total)
```

**Almost all originate from 2 nodes**: `bronze_distance_km_mean` and `bronze_duration_mean`

---

## Root Cause Summary

| Issue | Mechanism | Test Cases Affected | Fix Difficulty |
|-------|-----------|---------------------|-----------------|
| **Hub dominance is correct** | Distance/duration ARE causal ancestors in pipeline | All 22 (helps some, hurts others) | Cannot "fix" - it's correct |
| **Missing fuel metrics** | Graph doesn't include fuel sensor reading nodes | 3-4 cases (fuel_sensor, extreme_values) | Add fuel layer edges |
| **Missing prediction metrics** | Graph doesn't model ML pipeline causality | 4 cases (prediction_drift, outliers) | Add ML layer edges |
| **Missing data quality metrics** | Duplicates/validation not modeled | 2 cases (aggregation, duplicate handling) | Add ETL layer edges |
| **Negative weights** | Suppress anti-correlated anomalies | ~3-4 cases (GPS nulls, etc.) | Consider weight thresholding |
| **Pattern priors overfit** | Assume distance affects everything | Cascades through all cases | Restrict patterns to verifiable relationships |

---

## Recommended Actions

### Priority 1: Add Missing Metric Layers (2+ weeks)
1. **Fuel metrics layer**: Model raw fuel sensor → aggregated fuel metrics
   - `raw_std_fuel_consumption_ecol` → `p95_fuel_per_100km`
   - Add ~5 edges for fuel computation path
   
2. **ML pipeline layer**: Model feature → prediction → error
   - Features → `silver_ml_prediction_mean` → `silver_ml_prediction_std`
   - Add ~8 edges for ML pipeline
   
3. **Data quality layer**: Model duplicate/validation detection
   - ETL rules → `bronze_duplicate_rows_removed`
   - Add ~3 edges for quality checks

### Priority 2: Refine Pattern Priors (1 week)
1. Restrict PATTERN 4 (distance/duration → computed) to ONLY:
   - Distance → speed metrics (valid)
   - Duration → ingestion metrics (valid)
   - Remove: Distance → fuel metrics (incorrect assumption)

2. Add PATTERN 6: Source anomalies don't cascade
   - If raw_X is anomalous but downstream is normal → raw_X is independent

### Priority 3: Handle Negative Weights (2-3 days)
1. Option A: Remove negative edges (keep only positive/causal)
2. Option B: Ignore weight direction in RCA (use absolute value)
3. Option C: Interpret as "confounding" and handle differently

### Priority 4: Validation (1 week)
1. Run evaluation with each change incrementally
2. Track impact on failing test cases
3. Ensure no regression on passing cases

---

## Expected Improvement

If all three issues are fixed:

| Action | Expected Impact | Reasoning |
|--------|-----------------|-----------|
| Add fuel metrics | +2-3% Top-5 accuracy | Fuel cases (3-4) would get direct paths |
| Add ML metrics | +3-4% Top-5 accuracy | Prediction cases (4) would be reachable |
| Add data quality | +2% Top-5 accuracy | Duplicate/aggregation cases (2) would be detected |
| Refine patterns | +1-2% Top-5 accuracy | Reduce noise from wrong distance→fuel edges |
| **Total projected** | **+8-11%** | **Target: 62-65% Top-5 Accuracy** |

---

## Files to Update

1. **Causal discovery notebook** (`causal_discovery_v3_scalable.py`)
   - Modify `generate_pattern_based_priors()` function
   - Add new patterns for fuel/ML/quality layers
   - Adjust tier assignments

2. **RCA scoring** (`rca_severity_based_ranking.py`)
   - Handle negative weight edges
   - Consider fallback for missing graph edges
   - Add weight normalization option

3. **Graph artifacts**
   - Regenerate with updated discovery logic
   - Create new downstream/upstream maps
   - Update edge files with new relationships

