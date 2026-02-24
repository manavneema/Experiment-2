# Databricks notebook source
# MAGIC %md
# MAGIC # Unified RCA Evaluation Across All Causal Discovery Algorithms
# MAGIC 
# MAGIC This notebook evaluates root cause analysis performance across:
# MAGIC - **Pipeline A**: PC-Based (directed/undirected edges)
# MAGIC - **Pipeline B**: GraphicalLasso-Based (undirected edges, partial correlations)
# MAGIC - **Pipeline C**: NOTEARS-Based (directed DAG, structural weights)
# MAGIC - **Pipeline D**: Hybrid PC→NOTEARS→Bootstrap (directed DAG, bootstrap-stable weights)
# MAGIC 
# MAGIC Supports both **filtered** (after human priors) and **raw** (before human priors) graphs.

# COMMAND ----------

import json
import pandas as pd
import numpy as np
from pathlib import Path
from pyspark.sql import functions as F
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# ===========================
# CONFIGURATION
# ===========================

# Base artifact path
BASE_PATH = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"

# Metrics table
METRICS_TABLE = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"

# ===========================
# GRAPH REGISTRY
# ===========================
# Master registry of all available graphs with their configurations.
# Each graph has a unique key and all metadata needed to load and evaluate it.
# File detection is automatic based on consistent naming structure.

GRAPH_REGISTRY = {
    # PC Algorithm Graphs
    "pc_filtered": {
        "name": "PC Algorithm (Filtered)",
        "algorithm": "PC_Based",
        "folder": "PC_Based",
        "edge_type": "mixed",  # Has both directed and undirected edges
        "weight_column": "abs_weight",
        "edge_file": "pc_causal_edges.csv",
        "upstream_map": "upstream_map.json",
        "downstream_map": "downstream_map.json",
        "baseline_stats": "baseline_stats.json",
        "traversal_method": "downstream",  # PC produces mostly directed edges
    },
    "pc_raw": {
        "name": "PC Algorithm (Raw)",
        "algorithm": "PC_Based",
        "folder": "PC_Based",
        "edge_type": "mixed",
        "weight_column": "weight",
        "edge_file": "pc_raw_edges.csv",
        "upstream_map": "raw_upstream_map.json",
        "downstream_map": "raw_downstream_map.json",
        "baseline_stats": "baseline_stats.json",
        "traversal_method": "downstream",
    },
    
    # GraphicalLasso Graphs
    "glasso_filtered": {
        "name": "GraphicalLasso (Filtered)",
        "algorithm": "Graphical_Lasso_Based",
        "folder": "Graphical_Lasso_Based",
        "edge_type": "undirected",
        "weight_column": "abs_partial_corr",
        "edge_file": "graphical_lasso_causal_edges.csv",
        "upstream_map": "upstream_map.json",  # For undirected, upstream=downstream=neighbors
        "downstream_map": "downstream_map.json",
        "baseline_stats": "baseline_stats.json",
        "traversal_method": "upstream",  # Undirected graphs use upstream traversal from anomalies
    },
    "glasso_raw": {
        "name": "GraphicalLasso (Raw)",
        "algorithm": "Graphical_Lasso_Based",
        "folder": "Graphical_Lasso_Based",
        "edge_type": "undirected",
        "weight_column": "abs_partial_corr",
        "edge_file": "graphical_lasso_raw_edges.csv",
        "upstream_map": "raw_upstream_map.json",
        "downstream_map": "raw_downstream_map.json",
        "baseline_stats": "baseline_stats.json",
        "traversal_method": "upstream",
    },
    
    # NOTEARS Graphs
    "notears_filtered": {
        "name": "NOTEARS (Filtered)",
        "algorithm": "NOTEARS_Based",
        "folder": "NOTEARS_Based",
        "edge_type": "directed",  # True DAG with structural coefficients
        "weight_column": "abs_weight",
        "edge_file": "notears_causal_edges.csv",
        "upstream_map": "upstream_map.json",
        "downstream_map": "downstream_map.json",
        "baseline_stats": "baseline_stats.json",
        "traversal_method": "downstream",  # DAGs use downstream traversal
    },
    "notears_raw": {
        "name": "NOTEARS (Raw)",
        "algorithm": "NOTEARS_Based",
        "folder": "NOTEARS_Based",
        "edge_type": "directed",
        "weight_column": "abs_weight",
        "edge_file": "notears_raw_edges.csv",
        "upstream_map": "raw_upstream_map.json",
        "downstream_map": "raw_downstream_map.json",
        "baseline_stats": "baseline_stats.json",
        "traversal_method": "downstream",
    },
    
    # ===========================
    # Hybrid PC→NOTEARS→Bootstrap (Pipeline D)
    # ===========================
    # Best of both worlds: PC skeleton + NOTEARS weights + Bootstrap stability
    # Features: Automatic redundancy detection, tier constraints, structural priors
    "hybrid_filtered": {
        "name": "Hybrid PC-NOTEARS-Bootstrap (Filtered)",
        "algorithm": "Hybrid_PC_NOTEARS_Bootstrap",
        "folder": "Hybrid_PC_NOTEARS_Bootstrap",
        "edge_type": "directed",  # True DAG with bootstrap-stable edges
        "weight_column": "abs_weight",
        "edge_file": "hybrid_causal_edges.csv",
        "upstream_map": "upstream_map.json",
        "downstream_map": "downstream_map.json",
        "baseline_stats": "baseline_stats.json",
        "traversal_method": "downstream",  # DAG uses downstream traversal
    },
}

# ===========================
# GRAPHS TO EVALUATE
# ===========================
# Set True/False for each graph you want to include in this evaluation run.
# Examples:
#   - Test single graph: Set only one to True
#   - Compare filtered vs raw: Set both variants of one algorithm to True
#   - Compare all algorithms: Set all filtered (or all raw) to True
#   - Full comparison: Set all 6 to True

GRAPHS_TO_EVALUATE = {
    # PC Algorithm
    "pc_filtered": False,
    "pc_raw": False,
    
    # GraphicalLasso
    "glasso_filtered": False,
    "glasso_raw": False,
    
    # NOTEARS
    "notears_filtered": False,
    "notears_raw": False,
    
    # Hybrid PC-NOTEARS-Bootstrap (Pipeline D) - NEW
    "hybrid_filtered": True,
}

# ===========================
# TEST SCENARIOS
# ===========================
# Test dates to evaluate (supports multiple test cases in single run)
TEST_DATES = ["2026-02-06", "2026-02-09", "2026-02-10", "2026-02-11"]

# ===========================
# ANOMALY DETECTION THRESHOLDS
# ===========================
Z_SCORE_THRESHOLD = 3.0   # Flag if |z-score| > threshold
IQR_MULTIPLIER = 1.5      # Flag if value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

# ===========================
# GRAPH TRAVERSAL PARAMETERS
# ===========================
MAX_DEPTH = 3             # Maximum BFS traversal depth
DECAY_FACTOR = 0.6        # Score decay per hop (closer anomalies weighted higher)
SELF_ANOMALY_BONUS = 2.0  # Base multiplier if candidate itself is anomalous

# ===========================
# SEVERITY WEIGHTING (Option 1 - CloudRanger/MonitorRank approach)
# ===========================
# When enabled, anomalies contribute to scores proportional to their severity (|z-score|).
# This prevents hub nodes from dominating rankings when their downstream anomalies are mild.
# Reference: CloudRanger (Wang et al., ESEC/FSE 2018), MonitorRank (Kim et al., KDD 2013)
USE_SEVERITY_WEIGHTING = True  # Set False to use uniform weighting (baseline)

# ===========================
# OUT-DEGREE NORMALIZATION (Option 2 - Hub Penalty)
# ===========================
# When enabled, candidate scores are divided by (1 + out_degree) to penalize hub nodes.
# Hub nodes naturally reach many anomalies just by having many connections, not because
# they are true root causes. This normalization levels the playing field.
# Reference: Adtributor (Bhagwan et al., NSDI 2014) uses similar normalization.
USE_OUT_DEGREE_NORMALIZATION = True  # Set False to disable hub penalty

# ===========================
# PRINT CONFIGURATION
# ===========================
active_graphs = [k for k, v in GRAPHS_TO_EVALUATE.items() if v]
print("="*70)
print("EVALUATION CONFIGURATION")
print("="*70)
print(f"\nGraphs to evaluate ({len(active_graphs)} selected):")
for g in active_graphs:
    print("\n")
    for k, v in GRAPH_REGISTRY[g].items():
            print(f"{k}: {v}")

print(f"\nTest dates: {TEST_DATES}")
print(f"Z-score threshold: {Z_SCORE_THRESHOLD}")
print(f"IQR multiplier: {IQR_MULTIPLIER}")
print(f"Max traversal depth: {MAX_DEPTH}")
print(f"Decay factor: {DECAY_FACTOR}")
print(f"Severity weighting: {'ENABLED (CloudRanger approach)' if USE_SEVERITY_WEIGHTING else 'DISABLED (uniform)'}")
print(f"Out-degree normalization: {'ENABLED (hub penalty)' if USE_OUT_DEGREE_NORMALIZATION else 'DISABLED'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ground Truth Definition

# COMMAND ----------

# ===========================
# GROUND TRUTH
# ===========================

# Ground truth definitions for each test case.
# Labels are determined by SEMANTIC ANALYSIS (what the metric directly measures),
# NOT by observed anomaly magnitudes. This is critical for valid evaluation.
# 
# NOTE: These test cases correspond to fault injection functions in fault_injection_logic.py
# Run the fault injection on raw trip data, pass through ETL, then evaluate here.

GROUND_TRUTH = {
    # ---------------------------------------------------------
    # CASE 1: Raw Data Quality - NULL Injection (unit_id)
    # ---------------------------------------------------------
    "case1_unit_id_nulls": {
        "description": "40% of unit_id values set to NULL",
        "true_roots": {
            "raw_null_count_unit_id",  # Has edge to bronze_null_primary_key_rows
        },
        "test_date": "2026-02-06"
    },
    
    # ---------------------------------------------------------
    # CASE 2: Raw Data Quality - Distance/GPS Issues
    # ---------------------------------------------------------
    "case2_distance_gps_nulls": {
        "description": "Mixed evaluation 35% of rows have null gps_coverage and start_lat/long values",
        "true_roots": {
            "raw_null_count_distance",
            "raw_null_count_start_longitude",
            "raw_distance_mean",  
        },
        "test_date": "2026-02-09"
    },
    
    # ---------------------------------------------------------
    # CASE 3: ML Layer - Fuel Sensor Drift
    # ---------------------------------------------------------
    "case3_fuel_sensor_drift": {
        "description": "15% of fuel readings show 2-3.5x drift (sensor calibration failure)",
        "true_roots": {
            "silver_ml_large_error_count",  # ML detects prediction errors from bad fuel
            "p95_fuel_per_100km",            # KPI affected by drift
        },
        "test_date": "2026-02-10"
    },
    
    # ---------------------------------------------------------
    # CASE 4: Temporal - Clock Synchronization Failure
    # ---------------------------------------------------------
    "case4_clock_skew": {
        "description": "Device clock sync failure: 10% future dates, 10% negative durations",
        "true_roots": {
            "raw_max_trip_end_ts",           # Shows future timestamps
            "bronze_rows_dropped_by_rules",  # Validation catches bad durations
        },
        "test_date": "2026-02-11"
    },
}

# Map test dates to their ground truth case
DATE_TO_CASE = {v["test_date"]: k for k, v in GROUND_TRUTH.items()}

print("\nGround Truth Cases:")
for case_name, case_info in GROUND_TRUTH.items():
    print(f"  {case_name} ({case_info['test_date']}): {len(case_info['true_roots'])} root causes")
    for root in case_info['true_roots']:
        print(f"    - {root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utility Functions

# COMMAND ----------

def load_graph_artifacts(graph_key: str) -> Tuple[dict, dict, dict, dict]:
    """
    Load all artifacts for a specific graph configuration.
    
    Args:
        graph_key: Key from GRAPH_REGISTRY (e.g., 'pc_filtered', 'notears_raw')
        
    Returns:
        Tuple of (baseline_stats, upstream_map, downstream_map, edge_weights)
    """
    config = GRAPH_REGISTRY[graph_key]
    folder = f"{BASE_PATH}/{config['folder']}"
    
    # Load baseline stats
    baseline_path = f"{folder}/{config['baseline_stats']}"
    baseline_stats = {}
    if Path(baseline_path).exists():
        baseline_stats = json.loads(Path(baseline_path).read_text())
    else:
        print(f"    ⚠️ Missing: {config['baseline_stats']}")
    
    # Load upstream map
    upstream_path = f"{folder}/{config['upstream_map']}"
    upstream_map = {}
    if Path(upstream_path).exists():
        upstream_map = json.loads(Path(upstream_path).read_text())
    else:
        print(f"    ⚠️ Missing: {config['upstream_map']}")
    
    # Load downstream map
    downstream_path = f"{folder}/{config['downstream_map']}"
    downstream_map = {}
    if Path(downstream_path).exists():
        downstream_map = json.loads(Path(downstream_path).read_text())
    else:
        print(f"    ⚠️ Missing: {config['downstream_map']}")
    
    # Load edge weights
    edge_path = f"{folder}/{config['edge_file']}"
    edge_weights = {}
    if Path(edge_path).exists():
        df = pd.read_csv(edge_path)
        weight_col = config['weight_column']
        
        for _, row in df.iterrows():
            parent = row.get('from')
            child = row.get('to')
            
            # Get weight from algorithm-specific column
            weight = row.get(weight_col, row.get('weight', 1.0))
            
            # Handle NaN/None weights
            if pd.isna(weight) or weight is None:
                weight = 1.0
            
            # Normalize to positive value
            weight = abs(float(weight))
            
            # Store edge weight
            edge_weights[(parent, child)] = weight
            
            # For undirected graphs, add reverse edge
            if config["edge_type"] == "undirected":
                edge_weights[(child, parent)] = weight
    else:
        print(f"    ⚠️ Missing: {config['edge_file']}")
    
    return baseline_stats, upstream_map, downstream_map, edge_weights


def validate_graph_files(graph_key: str) -> bool:
    """Check if all required files exist for a graph configuration."""
    config = GRAPH_REGISTRY[graph_key]
    folder = f"{BASE_PATH}/{config['folder']}"
    
    required_files = [
        config['baseline_stats'],
        config['upstream_map'],
        config['downstream_map'],
        config['edge_file']
    ]
    
    missing = []
    for f in required_files:
        if not Path(f"{folder}/{f}").exists():
            missing.append(f)
    
    if missing:
        print(f"  ⚠️ {graph_key}: Missing files - {missing}")
        return False
    return True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Anomaly Detection
# MAGIC 
# MAGIC **Method**: Dual-test approach using Z-score AND IQR for robustness.
# MAGIC - **Z-score test**: Flags values > 3 standard deviations from mean (assumes normality)
# MAGIC - **IQR test**: Flags values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] (distribution-free)
# MAGIC - **Combined**: Anomaly if EITHER test triggers (more sensitive)
# MAGIC 
# MAGIC **Why dual tests?**
# MAGIC - Z-score catches extreme values in normally distributed metrics
# MAGIC - IQR catches outliers even in skewed distributions
# MAGIC - Together they provide better coverage across different metric distributions

# COMMAND ----------

def detect_anomalies(
    new_run: dict, 
    baseline_stats: dict, 
    z_thresh: float = 3.0, 
    iqr_multiplier: float = 1.5
) -> dict:
    """
    Detect anomalies using dual Z-score and IQR-based tests.
    
    Algorithm:
    1. For each metric in new_run, check if baseline stats exist
    2. Compute z-score: z = (value - mean) / std
    3. Compute IQR bounds: [Q1 - k*IQR, Q3 + k*IQR]
    4. Flag as anomaly if |z| > threshold OR value outside IQR bounds
    
    Args:
        new_run: Dict mapping metric_name -> observed value
        baseline_stats: Dict mapping metric_name -> {mean, std, q1, q3, IQR, n}
        z_thresh: Z-score threshold (default 3.0 = 99.7% of normal dist)
        iqr_multiplier: IQR multiplier for bounds (default 1.5 = Tukey's rule)
        
    Returns:
        Dict mapping anomalous_metric -> {value, z_score, z_flag, iqr_flag, ...}
    """
    anomalies = {}
    
    for metric, value in new_run.items():
        # Skip non-numeric values
        try:
            v = float(value)
        except (ValueError, TypeError):
            continue
        
        # Check if baseline exists with sufficient data
        stats = baseline_stats.get(metric)
        if not stats or stats.get('n', 0) < 5:  # Require at least 5 samples
            continue
        
        # Extract baseline statistics
        mean = stats.get('mean')
        std = stats.get('std') or 0.0
        q1 = stats.get('q1')
        q3 = stats.get('q3')
        iqr = stats.get('IQR') or 0.0
        
        # Compute z-score test
        z_score = None
        z_flag = False
        if std > 1e-10:  # Avoid division by near-zero std
            z_score = (v - mean) / std
            z_flag = abs(z_score) > z_thresh
        
        # Compute IQR-based outlier test
        iqr_flag = False
        lower_bound = None
        upper_bound = None
        if iqr > 1e-10 and q1 is not None and q3 is not None:
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            iqr_flag = (v < lower_bound) or (v > upper_bound)
        
        # Flag anomaly if EITHER test triggers
        if z_flag or iqr_flag:
            anomalies[metric] = {
                'value': float(v),
                'z_score': float(z_score) if z_score is not None else None,
                'z_flag': bool(z_flag),
                'iqr_flag': bool(iqr_flag),
                'lower_iqr': float(lower_bound) if lower_bound is not None else None,
                'upper_iqr': float(upper_bound) if upper_bound is not None else None,
                'baseline_mean': float(mean),
                'baseline_std': float(std),
                'direction': 'high' if (z_score and z_score > 0) else 'low'
            }
    
    return anomalies

# COMMAND ----------

# MAGIC %md
# MAGIC ## Candidate Scoring Algorithms
# MAGIC 
# MAGIC Two traversal methods are supported, each suited to different graph types:
# MAGIC 
# MAGIC ### 1. Downstream Traversal (for DAGs: NOTEARS, PC)
# MAGIC **Intuition**: A true root cause will have many anomalous metrics DOWNSTREAM of it.
# MAGIC - Start at each candidate metric
# MAGIC - BFS traverse following causal edges (parent → child)
# MAGIC - Score = sum of reachable anomalies weighted by decay and edge strength
# MAGIC - Bonus if candidate itself is anomalous
# MAGIC 
# MAGIC ### 2. Upstream Traversal (for undirected graphs: GraphicalLasso)
# MAGIC **Intuition**: Trace back from anomalies to find common upstream causes.
# MAGIC - Start at each detected anomaly
# MAGIC - BFS traverse following edges backward (child → parent)
# MAGIC - Metrics visited by MANY anomalies get higher scores
# MAGIC - Edge weights amplify scores along strong connections
# MAGIC 
# MAGIC ### Edge Weights
# MAGIC - **PC**: OLS regression coefficients (|β|)
# MAGIC - **GraphicalLasso**: Absolute partial correlations
# MAGIC - **NOTEARS**: Structural equation coefficients (|W_ij|)

# COMMAND ----------

def score_candidates_downstream(
    anomalies: dict,
    downstream_map: dict,
    edge_weights: dict = None,
    max_depth: int = 3,
    decay: float = 0.6,
    self_anomaly_bonus: float = 2.0,
    use_severity_weighting: bool = True,
    use_out_degree_norm: bool = True
) -> Tuple[dict, dict]:
    """
    Score root cause candidates by counting downstream anomalies they can reach.
    
    Algorithm (for each candidate):
    1. BFS from candidate following downstream edges
    2. At each hop, multiply score by decay factor and edge weight
    3. Accumulate score for each reachable anomaly, weighted by severity
    4. Apply bonus if candidate itself is anomalous (scaled by own severity)
    5. NORMALIZE by out-degree to penalize hub nodes
    
    Severity Weighting (CloudRanger/MonitorRank approach):
    - Each anomaly contributes score proportional to |z-score|
    - More severe anomalies have more influence on root cause ranking
    - Self-anomaly bonus is scaled by candidate's own severity
    
    Out-Degree Normalization (Adtributor approach):
    - Final score divided by (1 + out_degree)
    - Prevents hub nodes from dominating just because they have many connections
    - A node with 10 children needs 10x the raw score to beat a node with 1 child
    
    Args:
        anomalies: Dict of detected anomalies {metric: {z_score, value, ...}}
        downstream_map: Dict mapping node -> list of children
        edge_weights: Dict mapping (parent, child) -> weight
        max_depth: Maximum BFS depth
        decay: Score decay per hop (0.6 = 60% of previous hop's score)
        self_anomaly_bonus: Base score multiplier if candidate is anomalous
        use_severity_weighting: If True, weight by |z-score| (default True)
        use_out_degree_norm: If True, normalize by out-degree (default True)
        
    Returns:
        Tuple of (candidate_scores dict, traversal_details dict)
    """
    anomalous_metrics = set(anomalies.keys())
    
    # Pre-compute severity weights for all anomalies
    # Severity = |z-score|, normalized to [1, max_severity] range
    severity_weights = {}
    for metric, info in anomalies.items():
        z = info.get('z_score')
        if z is not None and use_severity_weighting:
            # Use |z-score| as severity, with minimum of 1.0
            severity_weights[metric] = max(1.0, abs(z))
        else:
            severity_weights[metric] = 1.0
    
    # Get all potential candidates
    all_candidates = set(anomalous_metrics)
    for metric in downstream_map.keys():
        all_candidates.add(metric)
    
    candidate_scores = {}
    traversal_details = {}
    
    for candidate in all_candidates:
        # BFS traversal downstream
        visited = set()
        queue = [(candidate, 0, 1.0)]  # (node, depth, accumulated_score)
        reachable_anomalies = []
        total_score = 0.0
        
        while queue:
            current_node, depth, score = queue.pop(0)
            
            if current_node in visited or depth > max_depth:
                continue
            
            visited.add(current_node)
            
            # Count reachable anomaly (excluding self)
            if current_node in anomalous_metrics and current_node != candidate:
                # SEVERITY WEIGHTING: Multiply by anomaly's severity
                severity = severity_weights.get(current_node, 1.0)
                weighted_contribution = score * severity
                reachable_anomalies.append((current_node, depth, score, severity, weighted_contribution))
                total_score += weighted_contribution
            
            # Expand to children
            if depth < max_depth:
                children = downstream_map.get(current_node, [])
                
                for child in children:
                    # Get edge weight (default 1.0 if not found)
                    weight = edge_weights.get((current_node, child), 1.0) if edge_weights else 1.0
                    # Propagate score with decay and edge weight
                    propagated_score = score * decay * weight
                    queue.append((child, depth + 1, propagated_score))
        
        candidate_scores[candidate] = total_score
        
        # Apply self-anomaly bonus (scaled by own severity if severity weighting enabled)
        if candidate in anomalous_metrics:
            own_severity = severity_weights.get(candidate, 1.0)
            # Dynamic bonus: base_bonus * (1 + log(severity)) for smoother scaling
            if use_severity_weighting and own_severity > 1.0:
                dynamic_bonus = self_anomaly_bonus * (1.0 + np.log(own_severity))
            else:
                dynamic_bonus = self_anomaly_bonus
            candidate_scores[candidate] *= dynamic_bonus
        
        # OUT-DEGREE NORMALIZATION: Penalize hub nodes
        # Divide by (1 + out_degree) so nodes with many children don't dominate
        out_degree = len(downstream_map.get(candidate, []))
        if use_out_degree_norm and out_degree > 0:
            candidate_scores[candidate] /= (1 + out_degree)
        
        # Store details for debugging/analysis
        traversal_details[candidate] = {
            'reachable_anomalies': reachable_anomalies,
            'is_anomalous': candidate in anomalous_metrics,
            'own_severity': severity_weights.get(candidate, None),
            'out_degree': out_degree,
            'raw_score': total_score,
            'final_score': candidate_scores[candidate]
        }
    
    return candidate_scores, traversal_details


def score_candidates_upstream(
    anomalies: dict,
    upstream_map: dict,
    edge_weights: dict = None,
    max_depth: int = 3,
    decay: float = 0.6,
    self_anomaly_bonus: float = 2.0,
    use_severity_weighting: bool = True,
    use_out_degree_norm: bool = True,
    downstream_map: dict = None
) -> Tuple[dict, dict]:
    """
    Score candidates by traversing upstream from each anomaly.
    
    Algorithm:
    1. For each anomaly, do BFS upstream (toward potential root causes)
    2. Each visited node accumulates score from all anomalies that reach it
    3. Scores decay with distance and scale with edge weights
    4. SEVERITY WEIGHTING: Each anomaly's contribution is weighted by |z-score|
    5. OUT-DEGREE NORMALIZATION: Divide by (1 + out_degree) to penalize hubs
    
    Severity Weighting (CloudRanger/MonitorRank approach):
    - Each anomaly contributes score proportional to |z-score|
    - More severe anomalies have more influence on root cause ranking
    - Self-anomaly bonus is scaled by candidate's own severity
    
    Out-Degree Normalization (Adtributor approach):
    - Final score divided by (1 + out_degree)
    - Prevents hub nodes from dominating just by having many connections
    
    Args:
        anomalies: Dict of detected anomalies {metric: {z_score, value, ...}}
        upstream_map: Dict mapping node -> list of parents
        edge_weights: Dict mapping (parent, child) -> weight
        max_depth: Maximum BFS depth
        decay: Score decay per hop
        self_anomaly_bonus: Base score multiplier if candidate is anomalous
        use_severity_weighting: If True, weight by |z-score| (default True)
        use_out_degree_norm: If True, normalize by out-degree (default True)
        downstream_map: Required for out-degree calculation
        
    Returns:
        Tuple of (candidate_scores dict, traversal_details dict)
    """
    anomalous_metrics = set(anomalies.keys())
    candidate_scores = defaultdict(float)
    contribution_details = defaultdict(list)  # Track which anomalies contribute to each candidate
    
    # Pre-compute severity weights for all anomalies
    severity_weights = {}
    for metric, info in anomalies.items():
        z = info.get('z_score')
        if z is not None and use_severity_weighting:
            severity_weights[metric] = max(1.0, abs(z))
        else:
            severity_weights[metric] = 1.0
    
    for anomalous_metric in anomalies.keys():
        # Get severity weight for this anomaly
        anomaly_severity = severity_weights.get(anomalous_metric, 1.0)
        
        # BFS traversal upstream from this anomaly
        visited = set()
        # Initial score is the anomaly's severity
        queue = [(anomalous_metric, 0, anomaly_severity)]
        
        while queue:
            current_node, depth, score = queue.pop(0)
            
            if current_node in visited or depth > max_depth:
                continue
            
            visited.add(current_node)
            
            # Accumulate score (already severity-weighted from initial value)
            candidate_scores[current_node] += score
            contribution_details[current_node].append({
                'from_anomaly': anomalous_metric,
                'anomaly_severity': anomaly_severity,
                'depth': depth,
                'contribution': score
            })
            
            # Expand to parents
            if depth < max_depth:
                parents = upstream_map.get(current_node, [])
                
                for parent in parents:
                    # Edge weight (note: edge is parent→current_node)
                    weight = edge_weights.get((parent, current_node), 1.0) if edge_weights else 1.0
                    propagated_score = score * decay * weight
                    queue.append((parent, depth + 1, propagated_score))
    
    # Apply self-anomaly bonus (scaled by own severity) and convert to regular dict
    final_scores = {}
    for candidate, score in candidate_scores.items():
        if candidate in anomalous_metrics:
            own_severity = severity_weights.get(candidate, 1.0)
            # Dynamic bonus: base_bonus * (1 + log(severity)) for smoother scaling
            if use_severity_weighting and own_severity > 1.0:
                dynamic_bonus = self_anomaly_bonus * (1.0 + np.log(own_severity))
            else:
                dynamic_bonus = self_anomaly_bonus
            final_scores[candidate] = score * dynamic_bonus
        else:
            final_scores[candidate] = score
    
    # OUT-DEGREE NORMALIZATION: Penalize hub nodes
    if use_out_degree_norm and downstream_map:
        for candidate in final_scores:
            out_degree = len(downstream_map.get(candidate, []))
            if out_degree > 0:
                final_scores[candidate] /= (1 + out_degree)
    
    # Build traversal details
    traversal_details = {}
    for candidate in final_scores:
        out_degree = len(downstream_map.get(candidate, [])) if downstream_map else 0
        traversal_details[candidate] = {
            'contributing_anomalies': contribution_details[candidate],
            'num_anomalies_reached': len(contribution_details[candidate]),
            'is_anomalous': candidate in anomalous_metrics,
            'own_severity': severity_weights.get(candidate, None),
            'out_degree': out_degree,
            'raw_score': candidate_scores[candidate],
            'final_score': final_scores[candidate]
        }
    
    return final_scores, traversal_details

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics

# COMMAND ----------

def compute_evaluation_metrics(predictions: list, true_roots: set) -> dict:
    """Compute all evaluation metrics."""
    
    def top_k_accuracy(preds, roots, k):
        return 1.0 if len(set(preds[:k]) & roots) > 0 else 0.0
    
    def mrr(preds, roots):
        for rank, pred in enumerate(preds, start=1):
            if pred in roots:
                return 1.0 / rank
        return 0.0
    
    def precision_at_k(preds, roots, k):
        tp = sum(1 for p in preds[:k] if p in roots)
        return tp / k if k > 0 else 0.0
    
    def recall_at_k(preds, roots, k):
        tp = len(set(preds[:k]) & roots)
        return tp / len(roots) if len(roots) > 0 else 0.0
    
    # Find ranks of true roots
    root_ranks = {}
    for root in true_roots:
        if root in predictions:
            root_ranks[root] = predictions.index(root) + 1
        else:
            root_ranks[root] = len(predictions) + 1  # Not found
    
    return {
        'top1_accuracy': top_k_accuracy(predictions, true_roots, 1),
        'top3_accuracy': top_k_accuracy(predictions, true_roots, 3),
        'top5_accuracy': top_k_accuracy(predictions, true_roots, 5),
        'top10_accuracy': top_k_accuracy(predictions, true_roots, 10),
        'mrr': mrr(predictions, true_roots),
        'precision@1': precision_at_k(predictions, true_roots, 1),
        'precision@3': precision_at_k(predictions, true_roots, 3),
        'precision@5': precision_at_k(predictions, true_roots, 5),
        'recall@3': recall_at_k(predictions, true_roots, 3),
        'recall@5': recall_at_k(predictions, true_roots, 5),
        'recall@10': recall_at_k(predictions, true_roots, 10),
        'root_ranks': root_ranks
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Test Data & Detect Anomalies

# COMMAND ----------

def load_test_run(test_date: str) -> dict:
    """Load metrics for a specific test date."""
    print(f"\nLoading test run for date: {test_date}")
    
    new_run_metrics = (
        spark.sql(f"SELECT * FROM {METRICS_TABLE} WHERE date = '{test_date}'")
        .select("date", "metric_name", "metric_value")
        .withColumn("metric_value", F.col("metric_value").cast("double"))
        .groupBy("date")
        .pivot("metric_name")
        .agg(F.first("metric_value"))
    )
    
    new_run_dict = new_run_metrics.drop('date').first().asDict()
    print(f"  Loaded {len(new_run_dict)} metrics")
    return new_run_dict

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Evaluation Loop

# COMMAND ----------

# ===========================
# MAIN EVALUATION LOOP
# ===========================

all_results = []
all_traversal_details = {}

# Get active graphs
active_graphs = [k for k, v in GRAPHS_TO_EVALUATE.items() if v]

if not active_graphs:
    print("⚠️ No graphs selected for evaluation!")
    print("   Set at least one graph to True in GRAPHS_TO_EVALUATE")
else:
    # Validate all selected graphs have required files
    print("\n" + "="*70)
    print("VALIDATING GRAPH FILES")
    print("="*70)
    valid_graphs = []
    for graph_key in active_graphs:
        if validate_graph_files(graph_key):
            valid_graphs.append(graph_key)
            print(f"  ✓ {graph_key}: All files present")
    
    if not valid_graphs:
        print("\n⚠️ No valid graphs found! Check file paths.")
    else:
        # Process each test date
        for test_date in TEST_DATES:
            print(f"\n{'='*70}")
            print(f"TEST DATE: {test_date}")
            print(f"{'='*70}")
            
            # Get ground truth for this date
            case_name = DATE_TO_CASE.get(test_date)
            if case_name:
                true_roots = GROUND_TRUTH[case_name]["true_roots"]
                print(f"Ground truth case: {case_name}")
                print(f"True root causes: {true_roots}")
            else:
                print(f"⚠️ No ground truth defined for {test_date}")
                true_roots = set()
            
            # Load test run metrics
            new_run_dict = load_test_run(test_date)
            
            # Evaluate each selected graph
            for graph_key in valid_graphs:
                config = GRAPH_REGISTRY[graph_key]
                
                print(f"\n{'-'*60}")
                print(f"GRAPH: {config['name']}")
                print(f"{'-'*60}")
                
                # Load artifacts
                baseline_stats, upstream_map, downstream_map, edge_weights = load_graph_artifacts(graph_key)
                
                print(f"  Edges loaded: {len(edge_weights)}")
                print(f"  Upstream nodes: {len(upstream_map)}")
                print(f"  Downstream nodes: {len(downstream_map)}")
                
                # Detect anomalies
                anomalies = detect_anomalies(
                    new_run_dict, 
                    baseline_stats, 
                    z_thresh=Z_SCORE_THRESHOLD, 
                    iqr_multiplier=IQR_MULTIPLIER
                )
                print(f"  Anomalies detected: {len(anomalies)}")
                
                if not anomalies:
                    print("  ⚠️ No anomalies detected - skipping scoring")
                    continue
                
                # Score candidates using appropriate traversal method
                traversal_method = config["traversal_method"]
                
                if traversal_method == "downstream":
                    scores, details = score_candidates_downstream(
                        anomalies, downstream_map, edge_weights,
                        max_depth=MAX_DEPTH, decay=DECAY_FACTOR,
                        self_anomaly_bonus=SELF_ANOMALY_BONUS,
                        use_severity_weighting=USE_SEVERITY_WEIGHTING,
                        use_out_degree_norm=USE_OUT_DEGREE_NORMALIZATION
                    )
                else:  # upstream
                    scores, details = score_candidates_upstream(
                        anomalies, upstream_map, edge_weights,
                        max_depth=MAX_DEPTH, decay=DECAY_FACTOR,
                        self_anomaly_bonus=SELF_ANOMALY_BONUS,
                        use_severity_weighting=USE_SEVERITY_WEIGHTING,
                        use_out_degree_norm=USE_OUT_DEGREE_NORMALIZATION,
                        downstream_map=downstream_map
                    )
                
                # Rank candidates
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                predictions = [m for m, s in ranked]
                
                # Evaluate against ground truth
                metrics = compute_evaluation_metrics(predictions, true_roots) if true_roots else {}
                
                # Store results
                result = {
                    'test_date': test_date,
                    'test_case': case_name,
                    'graph_key': graph_key,
                    'graph_name': config['name'],
                    'algorithm': config['algorithm'],
                    'graph_type': 'filtered' if 'filtered' in graph_key else 'raw',
                    'edge_type': config['edge_type'],
                    'traversal_method': traversal_method,
                    'num_anomalies': len(anomalies),
                    'num_edges': len(edge_weights),
                    'num_candidates': len(scores),
                    **metrics
                }
                all_results.append(result)
                
                # Store traversal details for debugging
                all_traversal_details[f"{test_date}_{graph_key}"] = details
                
                # Print top-10 candidates
                print(f"\n  Top-10 Candidates ({traversal_method} traversal):")
                for i, (metric, score) in enumerate(ranked[:10], 1):
                    is_root = "✓ ROOT" if metric in true_roots else ""
                    is_anom = "*" if metric in anomalies else ""
                    print(f"    {i:2d}. {metric:<42} (score={score:.3f}) {is_anom}{is_root}")
                
                # Print evaluation summary
                if metrics:
                    print(f"\n  Evaluation Summary:")
                    print(f"    Top-3 Accuracy: {metrics.get('top3_accuracy', 0):.3f}")
                    print(f"    MRR: {metrics.get('mrr', 0):.3f}")
                    print(f"    Recall@5: {metrics.get('recall@5', 0):.3f}")
                    for root, rank in metrics.get('root_ranks', {}).items():
                        status = "✓" if rank <= 3 else "⚠️" if rank <= 10 else "✗"
                        print(f"    {status} {root}: Rank {rank}")

# COMMAND ----------

print(f"\n✓ Evaluation complete: {len(all_results)} graph-date combinations evaluated")

# COMMAND ----------

# ===========================
# INDIVIDUAL TEST CASE SUMMARY TABLE
# ===========================

if all_results:
    print("\n" + "="*120)
    print("INDIVIDUAL TEST CASE RESULTS")
    print("="*120)
    
    # Build summary table
    summary_rows = []
    for result in all_results:
        # Get root ranks for display
        root_ranks = result.get('root_ranks', {})
        best_rank = min(root_ranks.values()) if root_ranks else 999
        roots_in_top3 = sum(1 for r in root_ranks.values() if r <= 3)
        roots_in_top5 = sum(1 for r in root_ranks.values() if r <= 5)
        total_roots = len(root_ranks)
        
        summary_rows.append({
            'Test Case': result.get('test_case', 'N/A'),
            'Date': result.get('test_date', 'N/A'),
            'Graph': result.get('graph_name', 'N/A').replace('Hybrid PC-NOTEARS-Bootstrap (Filtered)', 'Hybrid'),
            'Anomalies': result.get('num_anomalies', 0),
            'Top-1': f"{result.get('top1_accuracy', 0):.0%}",
            'Top-3': f"{result.get('top3_accuracy', 0):.0%}",
            'Top-5': f"{result.get('top5_accuracy', 0):.0%}",
            'MRR': f"{result.get('mrr', 0):.3f}",
            'P@3': f"{result.get('precision@3', 0):.3f}",
            'R@5': f"{result.get('recall@5', 0):.3f}",
            'Best Rank': best_rank,
            'Roots in Top-3': f"{roots_in_top3}/{total_roots}",
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))
    
    # ===========================
    # AGGREGATED METRICS
    # ===========================
    print("\n" + "="*120)
    print("AGGREGATED EVALUATION METRICS")
    print("="*120)
    
    results_df = pd.DataFrame(all_results)
    
    # Per-graph aggregates
    if len(results_df) > 0:
        agg_stats = {
            'Total Test Cases': len(results_df),
            'Mean Top-1 Accuracy': f"{results_df['top1_accuracy'].mean():.1%}",
            'Mean Top-3 Accuracy': f"{results_df['top3_accuracy'].mean():.1%}",
            'Mean Top-5 Accuracy': f"{results_df['top5_accuracy'].mean():.1%}",
            'Mean MRR': f"{results_df['mrr'].mean():.3f}",
            'Mean Precision@3': f"{results_df['precision@3'].mean():.3f}",
            'Mean Recall@5': f"{results_df['recall@5'].mean():.3f}",
            'Std MRR': f"{results_df['mrr'].std():.3f}" if len(results_df) > 1 else 'N/A',
        }
        
        print("\n")
        for metric, value in agg_stats.items():
            print(f"  {metric}: {value}")
        
        # Count successes
        top1_success = (results_df['top1_accuracy'] > 0).sum()
        top3_success = (results_df['top3_accuracy'] > 0).sum()
        top5_success = (results_df['top5_accuracy'] > 0).sum()
        
        print(f"\n  Success Rates:")
        print(f"    Root in Top-1: {top1_success}/{len(results_df)} ({100*top1_success/len(results_df):.0f}%)")
        print(f"    Root in Top-3: {top3_success}/{len(results_df)} ({100*top3_success/len(results_df):.0f}%)")
        print(f"    Root in Top-5: {top5_success}/{len(results_df)} ({100*top5_success/len(results_df):.0f}%)")
    
    # Per-test-case breakdown if multiple
    if len(results_df) > 1 and 'test_case' in results_df.columns:
        print("\n" + "-"*80)
        print("PER-FAULT-TYPE BREAKDOWN")
        print("-"*80)
        
        case_summary = results_df.groupby('test_case').agg({
            'mrr': ['mean', 'std'],
            'top3_accuracy': 'mean',
            'recall@5': 'mean',
            'num_anomalies': 'mean'
        }).round(3)
        
        print("\n" + case_summary.to_string())
else:
    print("\n⚠️ No results to summarize")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparison Summary

# COMMAND ----------

# ===========================
# COMPARISON SUMMARY
# ===========================

if all_results:
    print("\n" + "="*100)
    print("COMPARISON SUMMARY ACROSS ALL EVALUATED GRAPHS")
    print("="*100)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results)
    
    # Display key metrics
    display_cols = [
        'graph_name', 'graph_type', 'traversal_method',
        'top3_accuracy', 'mrr', 'precision@3', 'recall@5', 'num_edges', 'num_anomalies'
    ]
    
    # Filter to columns that exist
    display_cols = [c for c in display_cols if c in comparison_df.columns]
    
    print("\n" + comparison_df[display_cols].to_string(index=False))
    
    # Find best performing configuration
    if 'mrr' in comparison_df.columns and not comparison_df['mrr'].isna().all():
        best_idx = comparison_df['mrr'].idxmax()
        best = comparison_df.loc[best_idx]
        
        print(f"\n{'='*100}")
        print(f"🏆 BEST CONFIGURATION: {best['graph_name']}")
        print(f"{'='*100}")
        print(f"  MRR: {best.get('mrr', 'N/A'):.3f}")
        print(f"  Top-3 Accuracy: {best.get('top3_accuracy', 'N/A'):.3f}")
        print(f"  Recall@5: {best.get('recall@5', 'N/A'):.3f}")
        print(f"  Traversal method: {best.get('traversal_method', 'N/A')}")
        print(f"  Number of edges: {best.get('num_edges', 'N/A')}")
    
    # Show per-algorithm comparison if multiple algorithms tested
    if len(comparison_df['algorithm'].unique()) > 1:
        print(f"\n{'='*100}")
        print("PER-ALGORITHM SUMMARY (Best of filtered/raw)")
        print("="*100)
        
        algo_summary = comparison_df.groupby('algorithm').agg({
            'mrr': 'max',
            'top3_accuracy': 'max',
            'recall@5': 'max',
            'num_edges': 'mean'
        }).round(3)
        
        print("\n" + algo_summary.to_string())
    
    # Show filtered vs raw comparison
    if len(comparison_df['graph_type'].unique()) > 1:
        print(f"\n{'='*100}")
        print("FILTERED vs RAW COMPARISON (averaged across algorithms)")
        print("="*100)
        
        type_summary = comparison_df.groupby('graph_type').agg({
            'mrr': 'mean',
            'top3_accuracy': 'mean',
            'recall@5': 'mean'
        }).round(3)
        
        print("\n" + type_summary.to_string())
else:
    print("No results to compare")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Results

# COMMAND ----------

# ===========================
# EXPORT RESULTS
# ===========================

if all_results:
    # Export comparison results as CSV
    output_csv = f"{BASE_PATH}/evaluation_comparison.csv"
    comparison_df.to_csv(output_csv, index=False)
    print(f"✓ Saved comparison CSV to: {output_csv}")
    
    # Export detailed JSON with all metadata
    detailed_results = {
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "test_dates": TEST_DATES,
        "graphs_evaluated": [GRAPH_REGISTRY[k]['name'] for k in GRAPHS_TO_EVALUATE if GRAPHS_TO_EVALUATE[k]],
        "ground_truth": {k: {"roots": list(v["true_roots"]), "date": v["test_date"]} 
                         for k, v in GROUND_TRUTH.items()},
        "parameters": {
            "z_score_threshold": Z_SCORE_THRESHOLD,
            "iqr_multiplier": IQR_MULTIPLIER,
            "max_depth": MAX_DEPTH,
            "decay_factor": DECAY_FACTOR,
            "self_anomaly_bonus": SELF_ANOMALY_BONUS
        },
        "results": all_results
    }
    
    json_path = f"{BASE_PATH}/evaluation_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    print(f"✓ Saved detailed JSON to: {json_path}")
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
else:
    print("⚠️ No results to export") 
