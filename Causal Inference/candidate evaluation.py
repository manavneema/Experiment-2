# Databricks notebook source
# MAGIC %md
# MAGIC # RCA Evaluation Methodology
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This document explains how we evaluate the Root Cause Analysis system:
# MAGIC 1. **Ground Truth** (Case 1: 40% unit_id -> NULL)
# MAGIC 2. **RCA Pipeline** (3-stage anomaly detection - graph traversal - ranking)
# MAGIC 3. **Evaluation Metrics** (Top-K Accuracy, MRR, Precision, Recall)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Part 1: Ground Truth — Direct Measurement Principle
# MAGIC
# MAGIC **Case 1 Fault**: Set 40% of `unit_id` values to NULL in raw input table
# MAGIC
# MAGIC **Ground Truth Labels** (is_root):
# MAGIC - `raw_null_count_unit_id` = **1** (directly counts NULL unit_ids)
# MAGIC - `raw_unique_units` = **1** (directly measures unique count drop)
# MAGIC - All other 35 metrics = **0** (downstream effects)
# MAGIC
# MAGIC **Why Only These 2?**
# MAGIC
# MAGIC Ground truth is determined by **semantic analysis**, not observed values. We ask:
# MAGIC > *"Which metrics DIRECTLY measure the exact fault we injected?"*
# MAGIC
# MAGIC - ✓ `raw_null_count_unit_id` measures the fault itself (NULL counter)
# MAGIC - ✓ `raw_unique_units` measures the fault's complement (unique count)
# MAGIC - ✗ `bronze_survival_rate` is a downstream effect (rows filtered due to NULLs)
# MAGIC - ✗ `silver_vehicle_info_join_miss_rate` is 2 steps removed (NULLs → join fails → miss rate)
# MAGIC
# MAGIC **Key Point**: Other metrics anomalize due to causal propagation, but they measure *symptoms*, not the *disease*. This distinction is critical for evaluating if the RCA system can identify **original faults** vs **cascading effects**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Part 2: RCA Pipeline Architecture
# MAGIC
# MAGIC ### 3-Stage Process
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │ Stage 1: Anomaly Detection (Notebook1)                      │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │ Input:  New pipeline run metrics (37 metrics × 1 run)       │
# MAGIC │ Method: Z-score & IQR tests vs baseline (44-day history)    │
# MAGIC │ Output: detected_anomalies.json (~10-15 anomalous metrics)  │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC               ↓
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │ Stage 2: Graph Traversal & Ranking (Notebook2)              │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │ Input:  Anomalies + Frozen causal graph (53 edges)          │
# MAGIC │ Method: BFS downstream traversal from ALL metrics           │
# MAGIC │         Score = # reachable anomalies × decay × edge_weight │
# MAGIC │ Output: root_cause_candidates.csv (all metrics ranked)      │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC               ↓
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │ Stage 3: Evaluation (Notebook3)                             │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │ Input:  Predictions + Ground truth labels                   │
# MAGIC │ Method: Compute Top-K accuracy, MRR, Precision@K, Recall@K  │
# MAGIC │ Output: evaluation_summary.json                             │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Stage 2 Details: Scoring Algorithm
# MAGIC
# MAGIC **Core Logic**: A true root cause is **causally upstream** of many anomalies.
# MAGIC
# MAGIC For EACH metric in the system:
# MAGIC
# MAGIC For EACH metric in the system:
# MAGIC
# MAGIC 1. **BFS Downstream Traversal**: Start at the candidate, follow causal edges downstream
# MAGIC 2. **Count Anomalies**: Track how many anomalous metrics are reached
# MAGIC 3. **Apply Decay**: Distant anomalies contribute less (decay = 0.8 per hop)
# MAGIC 4. **Edge Weights**: Stronger causal edges (lower Granger p-values) → higher scores
# MAGIC 5. **Self-Anomaly Bonus**: If candidate itself is anomalous → score × 2.0
# MAGIC
# MAGIC **Example for Case 1**:
# MAGIC - `raw_null_count_unit_id` → reaches 8 downstream anomalies → score ≈ 6.4
# MAGIC - `bronze_survival_rate` → reaches 3 downstream anomalies → score ≈ 2.4
# MAGIC - `mean_fuel_per_100km` → reaches 0 downstream anomalies → score = 0.0
# MAGIC
# MAGIC **Why This Works**: Root causes naturally score higher because they're **structurally upstream** in the causal DAG, not because we hardcode assumptions about metric names or layers.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Part 3: Evaluation Metrics
# MAGIC
# MAGIC ### 1. Top-K Accuracy (Binary Hit Metric)
# MAGIC
# MAGIC ```python
# MAGIC Top-K Accuracy = 1.0 if ANY true root appears in top-K predictions, else 0.0
# MAGIC ```
# MAGIC
# MAGIC **What It Measures**: Did we find at least one root cause in the top-K?
# MAGIC
# MAGIC **Case 1 Targets**:
# MAGIC - Top-1: 0.5-1.0 (at least one root at rank 1)
# MAGIC - Top-3: 1.0 (both roots in top-3)
# MAGIC - Top-5: 1.0
# MAGIC
# MAGIC ### 2. Mean Reciprocal Rank (MRR)
# MAGIC
# MAGIC ```python
# MAGIC MRR = 1 / (rank of first true root cause)
# MAGIC ```
# MAGIC
# MAGIC **What It Measures**: How quickly do we encounter the first root cause?
# MAGIC
# MAGIC **Interpretation**:
# MAGIC - MRR = 1.0 → First root at rank 1 (perfect)
# MAGIC - MRR = 0.5 → First root at rank 2 (good)
# MAGIC - MRR = 0.33 → First root at rank 3 (acceptable)
# MAGIC - MRR = 0.1 → First root at rank 10 (poor)
# MAGIC
# MAGIC **Case 1 Target**: MRR ≥ 0.5
# MAGIC
# MAGIC ### 3. Precision@K
# MAGIC
# MAGIC ```python
# MAGIC Precision@K = (# true roots in top-K) / K
# MAGIC ```
# MAGIC
# MAGIC **What It Measures**: What fraction of top-K predictions are correct?
# MAGIC
# MAGIC **Case 1 Example** (2 true roots):
# MAGIC - Top-5 contains both roots → Precision@5 = 2/5 = 0.4
# MAGIC - Top-3 contains one root → Precision@3 = 1/3 = 0.33
# MAGIC
# MAGIC ### 4. Recall@K
# MAGIC
# MAGIC ```python
# MAGIC Recall@K = (# true roots in top-K) / (total # true roots)
# MAGIC ```
# MAGIC
# MAGIC **What It Measures**: What fraction of all true roots did we find?
# MAGIC
# MAGIC **Case 1 Example** (2 true roots):
# MAGIC - Top-5 contains both → Recall@5 = 2/2 = 1.0 (found all)
# MAGIC - Top-5 contains one → Recall@5 = 1/2 = 0.5 (found half)
# MAGIC - Top-5 contains zero → Recall@5 = 0/2 = 0.0 (found none)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Part 4: Success Criteria & Interpretation
# MAGIC
# MAGIC ## Part 4: Success Criteria & Interpretation
# MAGIC
# MAGIC ### Minimum Acceptance Thresholds
# MAGIC
# MAGIC | Metric | Minimum | Target | Failure Indicates |
# MAGIC |--------|---------|--------|-------------------|
# MAGIC | **Top-3 Accuracy** | **0.5** | 1.0 | Root cause not in top-3 → causal graph quality issue |
# MAGIC | **MRR** | **0.33** | 1.0 | First root beyond rank 3 → poor discrimination |
# MAGIC | **Precision@3** | 0.33 | 0.67 | Too many false positives in top-3 |
# MAGIC | **Recall@10** | 1.0 | 1.0 | Missing root causes entirely → detection failure |
# MAGIC
# MAGIC **Decision Rule**: If Top-3 Accuracy < 0.5, expand baseline from 44 to 60-90 days.
# MAGIC
# MAGIC ### What Success Looks Like (Ideal Output)
# MAGIC
# MAGIC ```
# MAGIC Top-10 Predictions vs Ground Truth:
# MAGIC ------------------------------------------------------------
# MAGIC  1. raw_null_count_unit_id              (score=6.8) ✓ TRUE ROOT
# MAGIC  2. raw_unique_units                    (score=5.2) ✓ TRUE ROOT
# MAGIC  3. bronze_null_primary_key_rows        (score=3.1) ✗
# MAGIC  4. bronze_survival_rate                (score=2.9) ✗
# MAGIC  5. silver_vehicle_info_join_miss_rate  (score=2.3) ✗
# MAGIC
# MAGIC EVALUATION RESULTS
# MAGIC ============================================================
# MAGIC Top-1 Accuracy:  1.000  ← Perfect! Root at #1
# MAGIC Top-3 Accuracy:  1.000  ← Both roots in top-3 ✓
# MAGIC MRR:             1.000  ← First root at rank 1
# MAGIC Precision@3:     0.667  ← 2/3 correct in top-3
# MAGIC Recall@10:       1.000  ← Found all roots
# MAGIC ```
# MAGIC
# MAGIC ### What Failure Looks Like (Poor Causal Graph)
# MAGIC
# MAGIC ```
# MAGIC Top-10 Predictions vs Ground Truth:
# MAGIC ------------------------------------------------------------
# MAGIC  1. bronze_distance_km_mean             (score=6.8) ✗
# MAGIC  2. raw_distance_mean                   (score=6.1) ✗
# MAGIC  3. bronze_duration_mean                (score=4.9) ✗
# MAGIC  6. raw_unique_units                    (score=3.9) ✓ TRUE ROOT
# MAGIC 10. raw_null_count_unit_id              (score=3.2) ✓ TRUE ROOT
# MAGIC
# MAGIC EVALUATION RESULTS
# MAGIC ============================================================
# MAGIC Top-1 Accuracy:  0.000  ← No root in top-1 ✗
# MAGIC Top-3 Accuracy:  0.000  ← No root in top-3 ✗ FAILURE
# MAGIC MRR:             0.167  ← First root at rank 6
# MAGIC Precision@3:     0.000  ← 0/3 correct in top-3
# MAGIC Recall@10:       1.000  ← Found all roots (but poorly ranked)
# MAGIC ```
# MAGIC
# MAGIC **Root Cause of Failure**: Causal graph has spurious edges or missing critical edges.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Part 5: Why This Evaluation is Valid
# MAGIC
# MAGIC ### 1. Fault-Agnostic Ground Truth
# MAGIC - Labels determined by **semantic meaning** (what the metric measures)
# MAGIC - NOT based on observed anomaly magnitudes
# MAGIC - Generalizes to any fault type (nulls, latency, duplicates, corruption)
# MAGIC
# MAGIC ### 2. Frozen Causal Graph (No Data Leakage)
# MAGIC - Graph trained on 44 days of **normal operation**
# MAGIC - Fault injection data NEVER seen during training
# MAGIC - Tests true causal inference, not memorization
# MAGIC
# MAGIC ### 3. Standard Information Retrieval Metrics
# MAGIC - Top-K Accuracy: Standard in search/recommendation systems
# MAGIC - MRR: Standard in ranking evaluation (IR, QA systems)
# MAGIC - Precision/Recall: Standard in classification/detection tasks
# MAGIC
# MAGIC ### 4. Interpretable & Actionable
# MAGIC - Can inspect which metrics ranked where and why
# MAGIC - Clear thresholds for success/failure
# MAGIC - Failure diagnosis points to specific fixes (expand data, tune parameters, add domain knowledge)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC **The Core Test**: Can the frozen causal graph correctly distinguish between:
# MAGIC - **Original faults** (raw layer DQ issues) 
# MAGIC - **Cascading effects** (downstream pipeline anomalies)
# MAGIC
# MAGIC This evaluation tests **causal structure learning quality**, not just anomaly detection. Success requires the PC algorithm to have learned the true causal relationships during the 44-day training period.
# MAGIC

# COMMAND ----------

import json
import pandas as pd
import numpy as np
from pathlib import Path
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# COMMAND ----------

# Defining configs
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"

ROOT_CAUSE_CANDIDATES = f"{path}/root_cause_candidates.csv"
# GROUND_TRUTH = ARTIFACT_DIR / 'ground_truth.csv'
# EVALUATION_OUT = ARTIFACT_DIR / 'evaluation_summary.json'

# COMMAND ----------

# DBTITLE 1,Cell 3
candidates_df = pd.read_csv(ROOT_CAUSE_CANDIDATES)
print(f"Loaded {len(candidates_df)} potential candidates (ranked)")

# Define ground truth (Case 1: 40% unit_id → NULL)
# Only metrics that DIRECTLY measure the injected fault are root causes
true_roots = {
    'raw_null_count_unit_id',  # Directly counts NULL values in unit_id
    'raw_unique_units'          # Directly measures unique unit_id drop
}

print(f"\nGround truth root causes: {len(true_roots)}")
for root in true_roots:
    print(f"  - {root}")

# COMMAND ----------

# Define evaluation metrics
def compute_top_k_accuracy(predictions, true_roots, k):
    """
    Compute Top-K accuracy.
    
    Args:
        predictions: List of predicted metrics (ranked)
        true_roots: Set of true root cause metrics
        k: Top-K to consider
        
    Returns:
        Float: 1.0 if any true root in top-K, else 0.0
    """
    top_k_preds = set(predictions[:k])
    hit = len(top_k_preds & true_roots) > 0
    return 1.0 if hit else 0.0


def compute_mean_reciprocal_rank(predictions, true_roots):
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        predictions: List of predicted metrics (ranked)
        true_roots: Set of true root cause metrics
        
    Returns:
        Float: MRR score
    """
    for rank, pred in enumerate(predictions, start=1):
        if pred in true_roots:
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(predictions, true_roots, k):
    """
    Compute Precision@K.
    
    Args:
        predictions: List of predicted metrics (ranked)
        true_roots: Set of true root cause metrics
        k: Top-K to consider
        
    Returns:
        Float: Precision@K
    """
    top_k_preds = predictions[:k]
    tp = sum(1 for pred in top_k_preds if pred in true_roots)
    return tp / k if k > 0 else 0.0


def compute_recall_at_k(predictions, true_roots, k):
    """
    Compute Recall@K.
    
    Args:
        predictions: List of predicted metrics (ranked)
        true_roots: Set of true root cause metrics
        k: Top-K to consider
        
    Returns:
        Float: Recall@K
    """
    top_k_preds = set(predictions[:k])
    tp = len(top_k_preds & true_roots)
    return tp / len(true_roots) if len(true_roots) > 0 else 0.0

# COMMAND ----------

# Extract ranked predictions
ranked_predictions = candidates_df['metric'].tolist()

# Compute metrics
top1_acc = compute_top_k_accuracy(ranked_predictions, true_roots, k=1)
top3_acc = compute_top_k_accuracy(ranked_predictions, true_roots, k=3)
top5_acc = compute_top_k_accuracy(ranked_predictions, true_roots, k=5)
top10_acc = compute_top_k_accuracy(ranked_predictions, true_roots, k=10)

mrr = compute_mean_reciprocal_rank(ranked_predictions, true_roots)

precision_at_1 = compute_precision_at_k(ranked_predictions, true_roots, k=1)
precision_at_3 = compute_precision_at_k(ranked_predictions, true_roots, k=3)
precision_at_5 = compute_precision_at_k(ranked_predictions, true_roots, k=5)
precision_at_10 = compute_precision_at_k(ranked_predictions, true_roots, k=10)

recall_at_1 = compute_recall_at_k(ranked_predictions, true_roots, k=1)
recall_at_3 = compute_recall_at_k(ranked_predictions, true_roots, k=3)
recall_at_5 = compute_recall_at_k(ranked_predictions, true_roots, k=5)
recall_at_10 = compute_recall_at_k(ranked_predictions, true_roots, k=10)

# Store results
evaluation_summary = {
    'top1_accuracy': float(top1_acc),
    'top3_accuracy': float(top3_acc),
    'top5_accuracy': float(top5_acc),
    'top10_accuracy': float(top10_acc),
    'mean_reciprocal_rank': float(mrr),
    'precision@1': float(precision_at_1),
    'precision@3': float(precision_at_3),
    'precision@5': float(precision_at_5),
    'precision@10': float(precision_at_10),
    'recall@1': float(recall_at_1),
    'recall@3': float(recall_at_3),
    'recall@5': float(recall_at_5),
    'recall@10': float(recall_at_10),
    'num_true_roots': len(true_roots),
    'num_predictions': len(ranked_predictions)
}

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"\nTop-1 Accuracy:  {top1_acc:.3f}")
print(f"Top-3 Accuracy:  {top3_acc:.3f}")
print(f"Top-5 Accuracy:  {top5_acc:.3f}")
print(f"Top-10 Accuracy: {top10_acc:.3f}")
print(f"\nMean Reciprocal Rank (MRR): {mrr:.3f}")
print(f"\nPrecision@1:  {precision_at_1:.3f}")
print(f"Precision@3:  {precision_at_3:.3f}")
print(f"Precision@5:  {precision_at_5:.3f}")
print(f"Precision@10: {precision_at_10:.3f}")
print(f"\nRecall@1:  {recall_at_1:.3f}")
print(f"Recall@3:  {recall_at_3:.3f}")
print(f"Recall@5:  {recall_at_5:.3f}")
print(f"Recall@10: {recall_at_10:.3f}")