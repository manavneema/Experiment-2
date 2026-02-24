# RCA Evaluation Methodology

## Overview

This document explains how we evaluate the Root Cause Analysis system:
1. **Ground Truth** (Case 1: 40% unit_id → NULL)
2. **RCA Pipeline** (3-stage anomaly detection → graph traversal → ranking)
3. **Evaluation Metrics** (Top-K Accuracy, MRR, Precision, Recall)

---

## Part 1: Ground Truth — Direct Measurement Principle

**Case 1 Fault**: Set 40% of `unit_id` values to NULL in raw input table

**Ground Truth Labels** (is_root):
- `raw_null_count_unit_id` = **1** (directly counts NULL unit_ids)
- `raw_unique_units` = **1** (directly measures unique count drop)
- All other 35 metrics = **0** (downstream effects)

**Why Only These 2?**

Ground truth is determined by **semantic analysis**, not observed values. We ask:
> *"Which metrics DIRECTLY measure the exact fault we injected?"*

- ✓ `raw_null_count_unit_id` measures the fault itself (NULL counter)
- ✓ `raw_unique_units` measures the fault's complement (unique count)
- ✗ `bronze_survival_rate` is a downstream effect (rows filtered due to NULLs)
- ✗ `silver_vehicle_info_join_miss_rate` is 2 steps removed (NULLs → join fails → miss rate)

**Key Point**: Other metrics anomalize due to causal propagation, but they measure *symptoms*, not the *disease*. This distinction is critical for evaluating if the RCA system can identify **original faults** vs **cascading effects**.

---

## Part 2: RCA Pipeline Architecture

## Part 2: RCA Pipeline Architecture

### 3-Stage Process

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Anomaly Detection (Notebook1)                      │
├─────────────────────────────────────────────────────────────┤
│ Input:  New pipeline run metrics (37 metrics × 1 run)       │
│ Method: Z-score & IQR tests vs baseline (44-day history)    │
│ Output: detected_anomalies.json (~10-15 anomalous metrics)  │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Graph Traversal & Ranking (Notebook2)              │
├─────────────────────────────────────────────────────────────┤
│ Input:  Anomalies + Frozen causal graph (53 edges)          │
│ Method: BFS downstream traversal from ALL metrics           │
│         Score = # reachable anomalies × decay × edge_weight │
│ Output: root_cause_candidates.csv (all metrics ranked)      │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Evaluation (Notebook3)                             │
├─────────────────────────────────────────────────────────────┤
│ Input:  Predictions + Ground truth labels                   │
│ Method: Compute Top-K accuracy, MRR, Precision@K, Recall@K  │
│ Output: evaluation_summary.json                             │
└─────────────────────────────────────────────────────────────┘
```

### Stage 2 Details: Scoring Algorithm

**Core Logic**: A true root cause is **causally upstream** of many anomalies.

For EACH metric in the system:

For EACH metric in the system:

1. **BFS Downstream Traversal**: Start at the candidate, follow causal edges downstream
2. **Count Anomalies**: Track how many anomalous metrics are reached
3. **Apply Decay**: Distant anomalies contribute less (decay = 0.8 per hop)
4. **Edge Weights**: Stronger causal edges (lower Granger p-values) → higher scores
5. **Self-Anomaly Bonus**: If candidate itself is anomalous → score × 2.0

**Example for Case 1**:
- `raw_null_count_unit_id` → reaches 8 downstream anomalies → score ≈ 6.4
- `bronze_survival_rate` → reaches 3 downstream anomalies → score ≈ 2.4
- `mean_fuel_per_100km` → reaches 0 downstream anomalies → score = 0.0

**Why This Works**: Root causes naturally score higher because they're **structurally upstream** in the causal DAG, not because we hardcode assumptions about metric names or layers.

---

## Part 3: Evaluation Metrics

### 1. Top-K Accuracy (Binary Hit Metric)

```python
Top-K Accuracy = 1.0 if ANY true root appears in top-K predictions, else 0.0
```

**What It Measures**: Did we find at least one root cause in the top-K?

**Case 1 Targets**:
- Top-1: 0.5-1.0 (at least one root at rank 1)
- Top-3: 1.0 (both roots in top-3)
- Top-5: 1.0

### 2. Mean Reciprocal Rank (MRR)

```python
MRR = 1 / (rank of first true root cause)
```

**What It Measures**: How quickly do we encounter the first root cause?

**Interpretation**:
- MRR = 1.0 → First root at rank 1 (perfect)
- MRR = 0.5 → First root at rank 2 (good)
- MRR = 0.33 → First root at rank 3 (acceptable)
- MRR = 0.1 → First root at rank 10 (poor)

**Case 1 Target**: MRR ≥ 0.5

### 3. Precision@K

```python
Precision@K = (# true roots in top-K) / K
```

**What It Measures**: What fraction of top-K predictions are correct?

**Case 1 Example** (2 true roots):
- Top-5 contains both roots → Precision@5 = 2/5 = 0.4
- Top-3 contains one root → Precision@3 = 1/3 = 0.33

### 4. Recall@K

```python
Recall@K = (# true roots in top-K) / (total # true roots)
```

**What It Measures**: What fraction of all true roots did we find?

**Case 1 Example** (2 true roots):
- Top-5 contains both → Recall@5 = 2/2 = 1.0 (found all)
- Top-5 contains one → Recall@5 = 1/2 = 0.5 (found half)
- Top-5 contains zero → Recall@5 = 0/2 = 0.0 (found none)

---

## Part 4: Success Criteria & Interpretation

## Part 4: Success Criteria & Interpretation

### Minimum Acceptance Thresholds

| Metric | Minimum | Target | Failure Indicates |
|--------|---------|--------|-------------------|
| **Top-3 Accuracy** | **0.5** | 1.0 | Root cause not in top-3 → causal graph quality issue |
| **MRR** | **0.33** | 1.0 | First root beyond rank 3 → poor discrimination |
| **Precision@3** | 0.33 | 0.67 | Too many false positives in top-3 |
| **Recall@10** | 1.0 | 1.0 | Missing root causes entirely → detection failure |

**Decision Rule**: If Top-3 Accuracy < 0.5, expand baseline from 44 to 60-90 days.

### What Success Looks Like (Ideal Output)

```
Top-10 Predictions vs Ground Truth:
------------------------------------------------------------
 1. raw_null_count_unit_id              (score=6.8) ✓ TRUE ROOT
 2. raw_unique_units                    (score=5.2) ✓ TRUE ROOT
 3. bronze_null_primary_key_rows        (score=3.1) ✗
 4. bronze_survival_rate                (score=2.9) ✗
 5. silver_vehicle_info_join_miss_rate  (score=2.3) ✗

EVALUATION RESULTS
============================================================
Top-1 Accuracy:  1.000  ← Perfect! Root at #1
Top-3 Accuracy:  1.000  ← Both roots in top-3 ✓
MRR:             1.000  ← First root at rank 1
Precision@3:     0.667  ← 2/3 correct in top-3
Recall@10:       1.000  ← Found all roots
```

### What Failure Looks Like (Poor Causal Graph)

```
Top-10 Predictions vs Ground Truth:
------------------------------------------------------------
 1. bronze_distance_km_mean             (score=6.8) ✗
 2. raw_distance_mean                   (score=6.1) ✗
 3. bronze_duration_mean                (score=4.9) ✗
 6. raw_unique_units                    (score=3.9) ✓ TRUE ROOT
10. raw_null_count_unit_id              (score=3.2) ✓ TRUE ROOT

EVALUATION RESULTS
============================================================
Top-1 Accuracy:  0.000  ← No root in top-1 ✗
Top-3 Accuracy:  0.000  ← No root in top-3 ✗ FAILURE
MRR:             0.167  ← First root at rank 6
Precision@3:     0.000  ← 0/3 correct in top-3
Recall@10:       1.000  ← Found all roots (but poorly ranked)
```

**Root Cause of Failure**: Causal graph has spurious edges or missing critical edges.

---

## Part 5: Why This Evaluation is Valid

### 1. Fault-Agnostic Ground Truth
- Labels determined by **semantic meaning** (what the metric measures)
- NOT based on observed anomaly magnitudes
- Generalizes to any fault type (nulls, latency, duplicates, corruption)

### 2. Frozen Causal Graph (No Data Leakage)
- Graph trained on 44 days of **normal operation**
- Fault injection data NEVER seen during training
- Tests true causal inference, not memorization

### 3. Standard Information Retrieval Metrics
- Top-K Accuracy: Standard in search/recommendation systems
- MRR: Standard in ranking evaluation (IR, QA systems)
- Precision/Recall: Standard in classification/detection tasks

### 4. Interpretable & Actionable
- Can inspect which metrics ranked where and why
- Clear thresholds for success/failure
- Failure diagnosis points to specific fixes (expand data, tune parameters, add domain knowledge)

---

## Conclusion

**The Core Test**: Can the frozen causal graph correctly distinguish between:
- **Original faults** (raw layer DQ issues) 
- **Cascading effects** (downstream pipeline anomalies)

This evaluation tests **causal structure learning quality**, not just anomaly detection. Success requires the PC algorithm to have learned the true causal relationships during the 44-day training period.
