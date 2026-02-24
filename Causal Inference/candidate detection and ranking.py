# Databricks notebook source
import json
import pandas as pd
import numpy as np
from pathlib import Path
from pyspark.sql import functions as F
from collections import defaultdict

# COMMAND ----------

# MAGIC %md
# MAGIC ## Defining configs

# COMMAND ----------

# Artifact path
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"

# Metrics table
metrics_table = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"

# Anomaly thresholds
Z_SCORE_THRESHOLD = 3.0
IQR_MULTIPLIER = 1.5

# Define paths
BASELINE_STATS = f"{path}/baseline_stats.json"
DETECTED_ANOMALIES_OUT = f"{path}/detected_anomalies.json"
UPSTREAM_MAP_PATH = f"{path}/upstream_map.json"
DOWNSTREAM_MAP_PATH = f"{path}/downstream_map.json"
GRANGER_EDGES = f"{path}/causal_candidates_granger.csv"
ROOT_CAUSE_OUT = f"{path}/root_cause_candidates.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Baseline Statistics

# COMMAND ----------

# DBTITLE 1,Cell 4
# Load baseline stats
assert Path(BASELINE_STATS).exists(), f"Run Notebook 1 first to generate: {BASELINE_STATS}"

baseline_stats = json.loads(Path(BASELINE_STATS).read_text())

print(f"Loaded baseline statistics for {len(baseline_stats)} metrics") 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detect Anomalies

# COMMAND ----------

def detect_anomalies(new_run, baseline_stats, z_thresh=3.0, iqr_multiplier=1.5):
    """
    Detect anomalies using z-score and IQR-based tests.
    
    Args:
        new_run: Dict mapping metric_name -> value
        baseline_stats: Dict mapping metric_name -> stats dict
        z_thresh: Z-score threshold for anomaly detection
        iqr_multiplier: IQR multiplier for outlier detection
        
    Returns:
        Dict mapping anomalous_metric -> anomaly details
    """
    anomalies = {}
    
    for metric, value in new_run.items():
        # Skip non-numeric values
        try:
            v = float(value)
        except (ValueError, TypeError):
            anomalies[metric] = {
                'value': value,
                'reason': 'non_numeric',
                'z_score': None,
                'iqr_flag': False,
                'z_flag': False
            }
            continue
        
        # Check if baseline exists
        stats = baseline_stats.get(metric)
        if not stats or stats.get('n', 0) == 0:
            anomalies[metric] = {
                'value': v,
                'reason': 'no_baseline',
                'z_score': None,
                'iqr_flag': False,
                'z_flag': False
            }
            continue
        
        # Extract baseline statistics
        mean = stats.get('mean')
        std = stats.get('std') or 0.0
        q1 = stats.get('q1')
        q3 = stats.get('q3')
        iqr = stats.get('IQR') or 0.0
        
        # Compute z-score
        z_score = None
        z_flag = False
        if std > 0:
            z_score = (v - mean) / std
            z_flag = abs(z_score) > z_thresh
        
        # Compute IQR-based outlier flag
        iqr_flag = False
        lower_bound = None
        upper_bound = None
        if iqr > 0:
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            iqr_flag = (v < lower_bound) or (v > upper_bound)
        
        # Flag anomaly if either test triggers
        if z_flag or iqr_flag:
            anomalies[metric] = {
                'value': float(v),
                'z_score': float(z_score) if z_score is not None else None,
                'z_flag': bool(z_flag),
                'iqr_flag': bool(iqr_flag),
                'lower_iqr': float(lower_bound) if lower_bound is not None else None,
                'upper_iqr': float(upper_bound) if upper_bound is not None else None,
                'baseline_mean': float(mean),
                'baseline_std': float(std)
            }
    
    return anomalies

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Downstream Traversal
# MAGIC ## Loading Causal Graph and Edge Weights

# COMMAND ----------

assert Path(DOWNSTREAM_MAP_PATH).exists(), f"Run casual discovery notebook first to generate: {DOWNSTREAM_MAP_PATH}"

downstream_map = json.loads(Path(DOWNSTREAM_MAP_PATH).read_text())

print(f"Loaded upstream map: {len(downstream_map)} nodes have parents")

# Load edge weights from Granger results
edge_weights = {}

granger_df = pd.read_csv(GRANGER_EDGES)

for _, row in granger_df.iterrows():
    parent = row.get('from')
    child = row.get('to')
    min_p = row.get('min_p', 1.0)
    
    # Convert p-value to weight: smaller p-value = stronger edge
    # weight = -log10(p) capped at 10
    weight = min(-np.log10(max(min_p, 1e-10)), 10.0)
    edge_weights[(parent, child)] = weight

print(f"\nLoaded edge weights for {len(edge_weights)} edges")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## Downstream Traversal and Candidate Scoring

# COMMAND ----------

## Scoring Root Cause Candidates
# **Scoring Logic**: For RCA, we need to find metrics that CAUSE many downstream anomalies:
# - Traverse **DOWNSTREAM** from each metric to count how many anomalies it can reach
# - Metrics that are upstream of many anomalies get higher scores
# - Add bonus if the metric itself is anomalous
# - **HUB NORMALIZATION**: Penalize high out-degree nodes to prevent hub bias
# - This naturally favors root causes without hardcoding assumptions
def score_root_cause_candidates_downstream(
    anomalies,
    downstream_map,
    edge_weights=None,
    max_depth=3,
    decay=0.8,
    self_anomaly_bonus=2.0,
    hub_normalize=True
):
    """
    Score root cause candidates by counting downstream anomalies they can reach.
    
    The intuition: A true root cause will have many anomalous metrics downstream.
    
    Hub Normalization: High out-degree nodes (hubs) naturally reach more nodes,
    so we normalize by log(1 + out_degree) to prevent hub bias.
    
    Args:
        anomalies: Dict of detected anomalies {metric: info}
        downstream_map: Dict mapping node -> list of child nodes
        edge_weights: Dict mapping (parent, child) -> weight
        max_depth: Maximum traversal depth
        decay: Score decay factor per hop
        self_anomaly_bonus: Bonus multiplier if candidate itself is anomalous
        hub_normalize: If True, apply hub normalization (score / log(1 + out_degree))
        
    Returns:
        Dict mapping candidate_metric -> score
    """
    anomalous_metrics = set(anomalies.keys())
    
    # Get all metrics that could be candidates (anomalous + their ancestors)
    all_candidates = set(anomalous_metrics)
    
    # Add all metrics that have any descendants (potential root causes)
    for metric in downstream_map.keys():
        all_candidates.add(metric)
    
    # Pre-compute out-degrees for hub normalization
    out_degrees = {node: len(children) for node, children in downstream_map.items()}
    
    candidate_scores = {}
    candidate_details = {}  # Store details for analysis
    
    for candidate in all_candidates:
        # BFS traversal downstream from this candidate
        visited = set()
        queue = [(candidate, 0, 1.0)]
        reachable_anomalies = set()
        total_score = 0.0
        
        while queue:
            current_node, depth, score = queue.pop(0)
            
            if current_node in visited or depth > max_depth:
                continue
            
            visited.add(current_node)
            
            # If this node is anomalous, count it
            if current_node in anomalous_metrics:
                reachable_anomalies.add(current_node)
                total_score += score
            
            # Expand to children
            if depth < max_depth:
                children = downstream_map.get(current_node, [])
                
                for child in children:
                    # Get edge weight
                    edge_weight = 1.0
                    if edge_weights:
                        edge_weight = edge_weights.get((current_node, child), 1.0)
                    
                    # Compute propagated score
                    propagated_score = score * decay * edge_weight
                    queue.append((child, depth + 1, propagated_score))
        
        # Raw score is the sum of reachable anomalies (with decay)
        raw_score = total_score
        
        # Bonus if the candidate itself is anomalous
        if candidate in anomalous_metrics:
            raw_score *= self_anomaly_bonus
        
        # Hub normalization: penalize high out-degree nodes
        out_degree = out_degrees.get(candidate, 0)
        if hub_normalize and out_degree > 0:
            # score / log(1 + degree) penalizes hubs without destroying influence
            normalized_score = raw_score / np.log(1 + out_degree)
        else:
            normalized_score = raw_score
        
        candidate_scores[candidate] = normalized_score
        candidate_details[candidate] = {
            'raw_score': raw_score,
            'out_degree': out_degree,
            'normalized_score': normalized_score,
            'reachable_anomalies': len(reachable_anomalies),
            'is_anomalous': candidate in anomalous_metrics
        }
    
    return candidate_scores

# COMMAND ----------

# MAGIC %md
# MAGIC # Upstream Traversal
# MAGIC ## Loading Causal Graph and Edge Weights

# COMMAND ----------

assert Path(UPSTREAM_MAP_PATH).exists(), f"Run casual discovery notebook first to generate: {UPSTREAM_MAP_PATH}"

upstream_map = json.loads(Path(UPSTREAM_MAP_PATH).read_text())

print(f"Loaded upstream map: {len(upstream_map)} nodes have parents")

# Load edge weights from Granger results
edge_weights = {}

granger_df = pd.read_csv(GRANGER_EDGES)

for _, row in granger_df.iterrows():
    parent = row.get('from')
    child = row.get('to')
    min_p = row.get('min_p', 1.0)
    
    # Convert p-value to weight: smaller p-value = stronger edge
    # weight = -log10(p) capped at 10
    weight = min(-np.log10(max(min_p, 1e-10)), 10.0)
    edge_weights[(parent, child)] = weight

print(f"\nLoaded edge weights for {len(edge_weights)} edges")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upstream Traversal and Candidate Scoring

# COMMAND ----------

def traverse_upstream_and_score(
    anomalies, 
    upstream_map,
    downstream_map=None,
    edge_weights=None,
    max_depth=3, 
    decay=0.6,
    hub_normalize=True
):
    """
    Traverse upstream from each anomalous metric and score root cause candidates.
    
    Hub Normalization: Nodes with many outgoing edges (high out-degree in downstream_map)
    are penalized to prevent hub bias.
    
    Args:
        anomalies: Dict of detected anomalies
        upstream_map: Dict mapping node -> list of parent nodes
        downstream_map: Dict mapping node -> list of child nodes (for hub normalization)
        edge_weights: Dict mapping (parent, child) -> weight
        max_depth: Maximum traversal depth
        decay: Score decay factor per hop
        hub_normalize: If True, apply hub normalization
        
    Returns:
        Dict mapping candidate_metric -> aggregated score
    """
    candidate_scores = defaultdict(float)
    traversal_details = []
    
    # Pre-compute out-degrees for hub normalization
    out_degrees = {}
    if downstream_map:
        out_degrees = {node: len(children) for node, children in downstream_map.items()}
    
    for anomalous_metric, anomaly_info in anomalies.items():
        # Initialize with score of 1.0 for each detected anomaly
        # (Do NOT use z-score magnitude - it biases toward high-variance metrics)
        initial_score = 1.0
        
        # BFS traversal upstream
        visited = set()
        queue = [(anomalous_metric, 0, initial_score)]
        
        while queue:
            current_node, depth, score = queue.pop(0)
            
            # Skip if already visited or max depth exceeded
            if current_node in visited or depth > max_depth:
                continue
            
            visited.add(current_node)
            
            # Accumulate score for this candidate
            candidate_scores[current_node] += score
            
            # Record traversal path
            traversal_details.append({
                'anomalous_metric': anomalous_metric,
                'candidate': current_node,
                'depth': depth,
                'score': score
            })
            
            # Expand to parents
            if depth < max_depth:
                parents = upstream_map.get(current_node, [])
                
                for parent in parents:
                    # Get edge weight
                    edge_weight = 1.0
                    if edge_weights:
                        edge_weight = edge_weights.get((parent, current_node), 1.0)
                    
                    # Compute propagated score with decay and edge weight
                    propagated_score = score * decay * edge_weight
                    
                    queue.append((parent, depth + 1, propagated_score))
    
    # Apply hub normalization after aggregation
    if hub_normalize and out_degrees:
        for candidate in candidate_scores:
            out_degree = out_degrees.get(candidate, 0)
            if out_degree > 0:
                candidate_scores[candidate] /= np.log(1 + out_degree)
    
    return dict(candidate_scores), traversal_details

# COMMAND ----------

# MAGIC %md
# MAGIC # Candidate Ranking Tests

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 1
# MAGIC - Loading Scenario 1 Run Metrics
# MAGIC - Running Anomoly Detection

# COMMAND ----------

# Try to load new run from CSV or JSON
new_run_metrics = (
    spark.sql(f"select * from {metrics_table} where date = '2026-01-16'")
    .select("date", "metric_name", "metric_value")
    .withColumn("metric_value", F.col("metric_value").cast("double"))
    .groupBy("date")
    .pivot("metric_name")
    .agg(F.first("metric_value"))
)

new_run_metrics = new_run_metrics.drop('date')

new_run_dic = new_run_metrics.first().asDict()

# Run anomaly detection
detected_anomalies = detect_anomalies(
    new_run_dic, 
    baseline_stats, 
    z_thresh=Z_SCORE_THRESHOLD, 
    iqr_multiplier=IQR_MULTIPLIER
)

print(f"\n{'='*60}")
print(f"Detected {len(detected_anomalies)} anomalous metrics.")
print(f"{'='*60}")

# Display detected anomalies
for metric, details in list(detected_anomalies.items())[:10]:
    print(f"\n{metric}:")
    print(f"  Value: {details.get('value')}")
    print(f"  Z-score: {details.get('z_score')}")
    print(f"  Z-flag: {details.get('z_flag')}, IQR-flag: {details.get('iqr_flag')}")

# COMMAND ----------

# Run scoring for downstream candidate analysis and ranking 
candidate_scores = score_root_cause_candidates_downstream(
    detected_anomalies,
    downstream_map,
    edge_weights=edge_weights,
    max_depth=MAX_DEPTH,
    decay=DECAY_FACTOR,
)

print(f"\n{'='*60}")
print(f"Scored {len(candidate_scores)} root cause candidates")
print(f"{'='*60}")

# Convert to DataFrame and rank
candidates_df = pd.DataFrame([
    {'metric': metric, 'score': score}
    for metric, score in candidate_scores.items()
])

# Sort by score (descending)
candidates_df = candidates_df.sort_values('score', ascending=False).reset_index(drop=True)
candidates_df['rank'] = candidates_df.index + 1

print("\nTop-10 Root Cause Candidates:")
print(candidates_df.head(10).to_string(index=False))

# COMMAND ----------

# Upstream Analysis
# Traversal parameters
MAX_DEPTH = 3
DECAY_FACTOR = 0.6

# Run traversal and scoring (with hub normalization)
candidate_scores, traversal_details = traverse_upstream_and_score(
    detected_anomalies,
    upstream_map,
    downstream_map=downstream_map,  # For hub normalization
    edge_weights=edge_weights,
    max_depth=MAX_DEPTH,
    decay=DECAY_FACTOR,
    hub_normalize=True
)

print(f"\n{'='*60}")
print(f"Scored {len(candidate_scores)} root cause candidates")
print(f"{'='*60}")
# Convert to DataFrame and rank
candidates_df = pd.DataFrame([
    {'metric': metric, 'score': score}
    for metric, score in candidate_scores.items()
])

# Sort by score (descending)
candidates_df = candidates_df.sort_values('score', ascending=False).reset_index(drop=True)
candidates_df['rank'] = candidates_df.index + 1

print("\nTop-10 Root Cause Candidates:")
display(candidates_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 2
# MAGIC - Loading Scenario 1 Run Metrics
# MAGIC - Running Anomoly Detection

# COMMAND ----------

# Try to load new run from CSV or JSON
new_run_metrics = (
    spark.sql(f"select * from {metrics_table} where date = '2026-01-17'")
    .select("date", "metric_name", "metric_value")
    .withColumn("metric_value", F.col("metric_value").cast("double"))
    .groupBy("date")
    .pivot("metric_name")
    .agg(F.first("metric_value"))
)

new_run_metrics = new_run_metrics.drop('date')

new_run_dic = new_run_metrics.first().asDict()

# Run anomaly detection
detected_anomalies_scenario2 = detect_anomalies(
    new_run_dic, 
    baseline_stats, 
    z_thresh=Z_SCORE_THRESHOLD, 
    iqr_multiplier=IQR_MULTIPLIER
)

print(f"\n{'='*60}")
print(f"Detected {len(detected_anomalies_scenario2)} anomalous metrics.")
print(f"{'='*60}")

# Display detected anomalies
for metric, details in list(detected_anomalies_scenario2.items())[:10]:
    print(f"\n{metric}:")
    print(f"  Value: {details.get('value')}")
    print(f"  Z-score: {details.get('z_score')}")
    print(f"  Z-flag: {details.get('z_flag')}, IQR-flag: {details.get('iqr_flag')}")

# COMMAND ----------

# Run scoring for downstream candidate analysis and ranking 
candidate_scores = score_root_cause_candidates_downstream(
    detected_anomalies_scenario2,
    downstream_map,
    edge_weights=edge_weights,
    max_depth=MAX_DEPTH,
    decay=DECAY_FACTOR,
)

print(f"\n{'='*60}")
print(f"Scored {len(candidate_scores)} root cause candidates")
print(f"{'='*60}")

# Convert to DataFrame and rank
candidates_df = pd.DataFrame([
    {'metric': metric, 'score': score}
    for metric, score in candidate_scores.items()
])

# Sort by score (descending)
candidates_df = candidates_df.sort_values('score', ascending=False).reset_index(drop=True)
candidates_df['rank'] = candidates_df.index + 1

print("\nTop-10 Root Cause Candidates:")
print(candidates_df.head(10).to_string(index=False))

# COMMAND ----------

# Upstream Analysis
# Traversal parameters
MAX_DEPTH = 3
DECAY_FACTOR = 0.6

# Run traversal and scoring (with hub normalization)
candidate_scores, traversal_details = traverse_upstream_and_score(
    detected_anomalies_scenario2,
    upstream_map,
    downstream_map=downstream_map,  # For hub normalization
    edge_weights=edge_weights,
    max_depth=MAX_DEPTH,
    decay=DECAY_FACTOR,
    hub_normalize=True
)

print(f"\n{'='*60}")
print(f"Scored {len(candidate_scores)} root cause candidates")
print(f"{'='*60}")
# Convert to DataFrame and rank
candidates_df = pd.DataFrame([
    {'metric': metric, 'score': score}
    for metric, score in candidate_scores.items()
])

# Sort by score (descending)
candidates_df = candidates_df.sort_values('score', ascending=False).reset_index(drop=True)
candidates_df['rank'] = candidates_df.index + 1

print("\nTop-10 Root Cause Candidates:")
display(candidates_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Code to Export Results
# MAGIC ## Runs once and writes the output of only one, either upstream analysis or downstream analysis

# COMMAND ----------

# Exporting upstream candidates
candidates_df.to_csv(ROOT_CAUSE_OUT, index=False)
print(f"\n✓ Saved root cause candidates to: {ROOT_CAUSE_OUT}")

print("✓ Notebook Complete — Root Causes Ranked")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Code Below to export detected anomalies

# COMMAND ----------

dbutils.fs.put(f"{path}/detected_anomalies.json", json.dumps(detected_anomalies, indent=2), overwrite=True)
print(f"✓ Saved upstream map to {path}/detected_anomalies.json")