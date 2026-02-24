# Databricks notebook source
# MAGIC %md
# MAGIC # Causal Discovery Utilities
# MAGIC 
# MAGIC Shared utility functions for causal discovery pipelines A, B, and C.
# MAGIC Contains common preprocessing, feature selection, visualization, and export functions.

# COMMAND ----------

"""
Causal Discovery Utilities Module

This module contains shared functions used across all three causal discovery pipelines:
- Pipeline A: PC-Based
- Pipeline B: GraphicalLasso-Based  
- Pipeline C: NOTEARS-Based

Functions are organized into categories:
- Constants (Human Priors)
- Data Loading
- Preprocessing
- Feature Selection
- Graph Utilities
- Statistics
"""

# COMMAND ----------

# Imports
import json
import numpy as np
import pandas as pd
from collections import defaultdict

from pyspark.sql import functions as F

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import networkx as nx
import matplotlib.pyplot as plt

# COMMAND ----------

# ===========================
# CONSTANTS
# ===========================

HUMAN_PRIOR_WHITELIST = [
    # ===========================================
    # RAW → BRONZE (Row count flow)
    # ===========================================
    ("raw_input_record_count", "bronze_input_rows"),          # Raw records become bronze input
    ("raw_null_count_unit_id", "bronze_null_primary_key_rows"),  # Null unit_ids → PK validation failures
    
    # ===========================================
    # RAW → BRONZE (Distribution propagation)
    # ===========================================
    ("raw_distance_mean", "bronze_distance_km_mean"),         # Distance distribution propagates
    ("raw_avg_speed_mean", "bronze_duration_mean"),           # Speed affects duration filtering
    ("raw_fuel_consumption_mean", "bronze_negative_fuel_events"),  # Fuel distribution → negative detection
    
    # ===========================================
    # BRONZE internal (Filtering logic)
    # ===========================================
    ("bronze_input_rows", "bronze_output_rows"),              # Output is filtered from input
    ("bronze_input_rows", "bronze_survival_rate"),            # Survival = output/input
    ("bronze_output_rows", "bronze_survival_rate"),           # Survival = output/input
    
    # ===========================================
    # BRONZE → SILVER (Row count flow)
    # ===========================================
    ("bronze_output_rows", "silver_input_data_count"),        # Bronze output IS silver input
    ("silver_input_data_count", "silver_count_after_feature_engineering"),  # FE transforms input
    ("silver_input_data_count", "silver_output_rows"),        # Output derived from input
    
    # ===========================================
    # BRONZE → SILVER (Survival rate propagation)
    # ===========================================
    ("bronze_survival_rate", "silver_survival_rate"),         # Upstream survival affects downstream
    
    # ===========================================
    # BRONZE → SILVER (Speed imputation - Notebook 3 lines 95-108)
    # avg_speed = distance / duration * 3.6
    # ===========================================
    ("bronze_distance_km_mean", "silver_avg_speed_imputed"),  # Distance used in speed calc
    ("bronze_duration_mean", "silver_avg_speed_imputed"),     # Duration used in speed calc
    
    # ===========================================
    # SILVER KPI calculations (Notebook 3 lines 170-230)
    # fuel_per_100km = (total_fuel / total_distance_km) * 100
    # idling_per_100km = (total_idle_time / total_distance_km) * 100
    # ===========================================
    ("bronze_distance_km_mean", "mean_fuel_per_100km"),       # Distance is denominator
    ("bronze_distance_km_mean", "p50_fuel_per_100km"),        # Distance is denominator
    ("bronze_distance_km_mean", "p95_fuel_per_100km"),        # Distance is denominator
    ("bronze_distance_km_mean", "mean_idling_per_100km"),     # Distance is denominator
    
    # ===========================================
    # SILVER ML model (Notebook 3 lines 130-165)
    # residual = fuel_consumption - prediction
    # abs_residual = abs(residual)
    # ===========================================
    ("silver_ml_prediction_mean", "silver_ml_residual_mean"),      # Residual depends on prediction
    ("silver_ml_residual_mean", "silver_ml_abs_residual_mean"),    # abs(residual) from residual
    ("silver_ml_residual_std", "silver_ml_abs_residual_p95"),      # Std affects p95
    ("silver_ml_imputed_fuel_mean", "silver_ml_imputed_fuel_p95"), # Mean affects distribution
]

# COMMAND ----------

# ===========================
# DATA LOADING
# ===========================

def spark_metrics_to_matrix(metrics_sdf, max_runs=65, date_col='date'):
    """
    Convert Spark metrics DataFrame to wide pandas DataFrame.
    
    Args:
        metrics_sdf: Spark DataFrame with columns [date, metric_name, metric_value]
        max_runs: Number of most recent dates to include
        date_col: Name of the date column
    
    Returns:
        pd.DataFrame: Wide format with dates as index, metrics as columns
    """
    sdf = metrics_sdf.select(
        F.col(date_col).alias('date'), 
        F.col('metric_name'), 
        F.col('metric_value')
    )
    
    # Keep max_runs most recent dates
    recent_dates = sdf.select('date').distinct().orderBy(F.desc('date')).limit(max_runs)
    recent = sdf.join(recent_dates, on='date', how='inner')
    
    # Pivot and aggregate
    pivot = (recent
             .withColumn('metric_value', F.col('metric_value').cast('double'))
             .groupBy('date')
             .pivot('metric_name')
             .agg(F.first('metric_value')))
    
    pdf = pivot.orderBy('date').toPandas()
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.set_index('date').sort_index()
    
    print(f"Loaded {len(pdf)} days of metrics data")
    print(f"Metrics available: {len(pdf.columns)}")
    
    return pdf

# COMMAND ----------

# ===========================
# PREPROCESSING
# ===========================

def generate_stage_blacklist(metric_cols):
    """
    Generate blacklist pairs enforcing pipeline flow direction.
    
    Rules: raw → bronze → silver (forbids reverse edges)
    
    Args:
        metric_cols: Iterable of metric column names
    
    Returns:
        list: List of (from, to) tuples to blacklist
    """
    blacklist = []
    cols = list(metric_cols)
    for a in cols:
        for b in cols:
            if a == b:
                continue
            if a.startswith('silver_') and (b.startswith('bronze_') or b.startswith('raw_')):
                blacklist.append((a, b))
            if a.startswith('bronze_') and b.startswith('raw_'):
                blacklist.append((a, b))
    return blacklist


def preprocess_metrics_matrix(
    df,
    zscore=True,
    impute_strategy="median",
    max_missing_frac=0.5,
    feature_sample_ratio=2.5,
    min_keep_features=20,
):
    """
    Preprocess metrics matrix for causal discovery.
    
    Args:
        df: Raw metrics DataFrame
        zscore: Whether to apply z-score normalization
        impute_strategy: Imputation strategy ('median', 'mean')
        max_missing_frac: Max fraction of missing values before dropping column
        feature_sample_ratio: Max features/samples ratio before reduction
        min_keep_features: Minimum features to keep after reduction
    
    Returns:
        tuple: (preprocessed DataFrame, metadata dict)
    """
    meta = {}
    
    # Normalize null tokens
    df = df.replace(["null", "NULL", "None", ""], np.nan)
    
    # Coerce to numeric
    df_num = df.copy()
    for c in df_num.columns:
        if not pd.api.types.is_numeric_dtype(df_num[c]):
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    
    meta["initial_shape"] = df_num.shape
    
    # Missingness stats
    na_mask = df_num.isna()
    miss_frac = na_mask.mean()
    meta["missing_fraction"] = miss_frac.to_dict()
    
    # Drop fully-null & high-missing columns
    drop_all_null = miss_frac[miss_frac == 1.0].index.tolist()
    drop_high_missing = miss_frac[miss_frac > max_missing_frac].index.tolist()
    drop_cols = sorted(set(drop_all_null + drop_high_missing))
    
    meta["dropped_all_null"] = drop_all_null
    meta["dropped_high_missing"] = drop_high_missing
    
    if drop_cols:
        df_num = df_num.drop(columns=drop_cols)
        na_mask = na_mask.drop(columns=drop_cols)
    
    # Drop constant columns
    nunique = df_num.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    meta["dropped_constant"] = const_cols
    
    if const_cols:
        df_num = df_num.drop(columns=const_cols)
        na_mask = na_mask.drop(columns=const_cols)
    
    if df_num.shape[1] == 0:
        meta["final_shape"] = (df_num.shape[0], 0)
        return pd.DataFrame(index=df.index), meta
    
    # Add missingness indicators
    partial_missing_cols = miss_frac[
        (miss_frac > 0.0) & (miss_frac <= max_missing_frac)
    ].index.intersection(df_num.columns)
    
    if len(partial_missing_cols) > 0:
        indicators = na_mask[partial_missing_cols].astype("float32")
        indicators.columns = [f"missing_{c}" for c in indicators.columns]
    else:
        indicators = None
    
    meta["num_missing_indicators"] = 0 if indicators is None else indicators.shape[1]
    
    # Impute numeric values
    imputer = SimpleImputer(strategy=impute_strategy)
    imputed_array = imputer.fit_transform(df_num.values)
    
    # Z-score scaling
    if zscore:
        scaler = StandardScaler()
        imputed_array = scaler.fit_transform(imputed_array)
    
    df_clean = pd.DataFrame(
        imputed_array,
        index=df_num.index,
        columns=df_num.columns,
    )
    
    # Feature reduction if features >> samples
    n_samples, n_feats = df_clean.shape
    meta["pre_reduction_shape"] = (n_samples, n_feats)
    
    if n_feats > n_samples * feature_sample_ratio:
        var = df_clean.var().sort_values(ascending=False)
        keep_k = max(int(n_samples * 1.5), min_keep_features)
        keep_cols = var.index[:keep_k]
        
        df_clean = df_clean[keep_cols]
        meta["feature_reduction"] = {
            "applied": True,
            "kept_features": len(keep_cols),
            "dropped_features": n_feats - len(keep_cols),
        }
    else:
        meta["feature_reduction"] = {"applied": False}
    
    # Reattach indicators (unscaled)
    if indicators is not None:
        df_clean = pd.concat([df_clean, indicators], axis=1)
    
    # Final checks
    assert not df_clean.isna().any().any(), "NaNs remain after preprocessing"
    
    meta["final_shape"] = df_clean.shape
    return df_clean, meta

# COMMAND ----------

# ===========================
# FEATURE SELECTION
# ===========================

def _get_feature_priority(feature_name):
    """
    Score feature importance for selection priority.
    
    Higher score = higher priority (keep feature).
    
    Args:
        feature_name: Name of the feature
    
    Returns:
        int: Priority score
    """
    score = 0
    
    # Prefer rates over raw counts
    if any(term in feature_name.lower() for term in ['rate', 'ratio', 'percent', 'pct']):
        score += 10
    
    # Prefer aggregated KPIs
    if any(term in feature_name.lower() for term in ['mean', 'avg', 'median', 'p95']):
        score += 5
    
    # Prefer downstream stages
    if feature_name.startswith('silver_'):
        score += 8
    elif feature_name.startswith('bronze_'):
        score += 4
    elif feature_name.startswith('raw_'):
        score += 0
    
    # Prefer business metrics
    if any(term in feature_name.lower() for term in ['fuel', 'speed', 'distance', 'trip']):
        score += 6
    
    # Prefer duration metrics
    if 'duration' in feature_name.lower():
        score += 3
    
    return score


def sophisticated_feature_selection(
    df,
    target_features=40,
    variance_threshold=1e-6,
    correlation_threshold=0.95,
):
    """
    Feature selection optimized for causal discovery algorithms.
    
    Args:
        df: Input DataFrame
        target_features: Target number of features to keep
        variance_threshold: Min variance to keep feature
        correlation_threshold: Max correlation before pruning
    
    Returns:
        tuple: (selected DataFrame, selection log dict)
    """
    print(f"Starting feature selection (target: {target_features} features)")
    print(f"Input features: {df.shape[1]}")
    
    selection_log = {
        "initial_features": df.shape[1],
        "target_features": target_features,
        "steps": []
    }
    
    out = df.copy()
    
    # Step 1: Variance Filter
    print("\n[Step 1] Variance filtering...")
    var = out.var()
    low_var_cols = var[var <= variance_threshold].index.tolist()
    
    if low_var_cols:
        out = out.drop(columns=low_var_cols)
        print(f"  Removed {len(low_var_cols)} near-constant features")
        selection_log["steps"].append({
            "step": "variance_filter",
            "removed_count": len(low_var_cols),
            "removed_features": low_var_cols,
            "reason": f"variance <= {variance_threshold}"
        })
    
    # Step 2: Redundant Metric Removal
    print("\n[Step 2] Redundant metric removal...")
    redundant_removed = []
    current_cols = list(out.columns)
    
    # Remove multiple percentiles (keep p95 only)
    percentile_groups = {}
    for col in current_cols:
        if any(p in col for p in ['_p25', '_p50', '_p75', '_p90', '_p99']):
            base_name = col.split('_p')[0]
            if base_name not in percentile_groups:
                percentile_groups[base_name] = []
            percentile_groups[base_name].append(col)
    
    for base_name, percentile_cols in percentile_groups.items():
        if any('_p95' in col for col in percentile_cols):
            keep_col = next(col for col in percentile_cols if '_p95' in col)
        else:
            keep_col = percentile_cols[-1]
        
        to_remove = [col for col in percentile_cols if col != keep_col]
        redundant_removed.extend(to_remove)
    
    # Remove duplicate row counts per stage
    row_count_groups = {}
    for col in current_cols:
        if any(term in col.lower() for term in ['rows', 'count', 'records']):
            if col.startswith('raw_'):
                stage = 'raw'
            elif col.startswith('bronze_'):
                stage = 'bronze'
            elif col.startswith('silver_'):
                stage = 'silver'
            else:
                stage = 'other'
            
            if stage not in row_count_groups:
                row_count_groups[stage] = []
            row_count_groups[stage].append(col)
    
    for stage, count_cols in row_count_groups.items():
        if len(count_cols) > 2:
            priority_patterns = ['input_rows', 'output_rows', 'record_count']
            prioritized = [col for col in count_cols if any(p in col for p in priority_patterns)]
            others = [col for col in count_cols if col not in prioritized]
            
            keep_cols = prioritized[:2] if len(prioritized) >= 2 else prioritized + others[:2-len(prioritized)]
            to_remove = [col for col in count_cols if col not in keep_cols]
            redundant_removed.extend(to_remove)
    
    if redundant_removed:
        out = out.drop(columns=redundant_removed)
        print(f"  Removed {len(redundant_removed)} redundant features")
        selection_log["steps"].append({
            "step": "redundant_removal",
            "removed_count": len(redundant_removed),
            "removed_features": redundant_removed,
            "reason": "structural_redundancy"
        })
    
    # Step 3: High-Correlation Pruning
    print("\n[Step 3] Correlation-based pruning...")
    if out.shape[1] > target_features:
        corr_matrix = out.corr().abs()
        
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
        
        correlation_removed = []
        remaining_cols = set(out.columns)
        
        for col1, col2, corr_val in sorted(corr_pairs, key=lambda x: x[2], reverse=True):
            if col1 in remaining_cols and col2 in remaining_cols:
                priority1 = _get_feature_priority(col1)
                priority2 = _get_feature_priority(col2)
                
                if priority1 >= priority2:
                    remove_col, keep_col = col2, col1
                else:
                    remove_col, keep_col = col1, col2
                
                correlation_removed.append(remove_col)
                remaining_cols.remove(remove_col)
                print(f"    Removed {remove_col} (corr={corr_val:.3f} with {keep_col})")
        
        if correlation_removed:
            out = out.drop(columns=correlation_removed)
            selection_log["steps"].append({
                "step": "correlation_pruning",
                "removed_count": len(correlation_removed),
                "removed_features": correlation_removed,
                "reason": f"correlation > {correlation_threshold}"
            })
    
    # Step 4: Final size adjustment
    print("\n[Step 4] Final size adjustment...")
    if out.shape[1] > target_features:
        feature_variances = out.var().sort_values(ascending=False)
        keep_features = feature_variances.head(target_features).index.tolist()
        final_removed = [col for col in out.columns if col not in keep_features]
        
        out = out[keep_features]
        print(f"  Final trim: removed {len(final_removed)} lowest-variance features")
        selection_log["steps"].append({
            "step": "final_variance_trim",
            "removed_count": len(final_removed),
            "removed_features": final_removed,
            "reason": f"variance_ranking_to_reach_target_{target_features}"
        })
    
    # Summary
    selection_log["final_features"] = out.shape[1]
    selection_log["total_removed"] = df.shape[1] - out.shape[1]
    selection_log["reduction_ratio"] = selection_log["total_removed"] / df.shape[1] if df.shape[1] > 0 else 0
    
    print(f"\n✓ Feature selection complete:")
    print(f"  Initial: {df.shape[1]} features")
    print(f"  Final: {out.shape[1]} features")
    print(f"  Removed: {selection_log['total_removed']} features ({selection_log['reduction_ratio']:.1%})")
    print(f"  Target achieved: {'✓' if out.shape[1] <= target_features else '✗'}")
    
    return out, selection_log

# COMMAND ----------

# ===========================
# GRAPH UTILITIES
# ===========================

def build_adjacency_maps(edges, handle_undirected=False):
    """
    Build upstream and downstream adjacency maps from edge list.
    
    Args:
        edges: List of edge dicts or tuples
        handle_undirected: If True, add both directions for undirected edges
    
    Returns:
        tuple: (upstream_map dict, downstream_map dict)
    """
    upstream_map = defaultdict(list)
    downstream_map = defaultdict(list)
    
    for edge in edges:
        if isinstance(edge, dict):
            parent, child = edge["from"], edge["to"]
            edge_type = edge.get("type", "directed")
        else:
            parent, child = edge[0], edge[1]
            edge_type = edge[2] if len(edge) > 2 else "directed"
        
        # Add directed edge
        upstream_map[child].append(parent)
        downstream_map[parent].append(child)
        
        # For undirected edges, add reverse direction
        if handle_undirected and edge_type == "undirected":
            upstream_map[parent].append(child)
            downstream_map[child].append(parent)
    
    return dict(upstream_map), dict(downstream_map)


def visualize_skeleton(edges, title="Causal Skeleton", top_k=50):
    """
    Visualize causal skeleton graph (for PC and GraphicalLasso).
    
    Args:
        edges: List of edge dicts or tuples
        title: Plot title
        top_k: Maximum edges to display
    
    Returns:
        nx.Graph: NetworkX graph object
    """
    if not edges:
        print("No edges to visualize")
        return None
    
    # Sort by weight if available and limit to top_k
    if len(edges) > top_k:
        try:
            if isinstance(edges[0], dict):
                weight_key = "abs_partial_corr" if "abs_partial_corr" in edges[0] else "abs_weight"
                sorted_edges = sorted(edges, key=lambda x: x.get(weight_key, 0), reverse=True)
            else:
                sorted_edges = edges
            show_edges = sorted_edges[:top_k]
        except:
            show_edges = edges[:top_k]
    else:
        show_edges = edges
    
    # Determine if graph has directed edges
    has_directed = False
    for edge in show_edges:
        if isinstance(edge, dict):
            if edge.get("type", "undirected") == "directed" or edge.get("edge_type", "undirected") == "directed":
                has_directed = True
                break
        elif len(edge) > 2 and edge[2] == "directed":
            has_directed = True
            break
    
    G = nx.DiGraph() if has_directed else nx.Graph()
    
    for edge in show_edges:
        if isinstance(edge, dict):
            a, b = edge["from"], edge["to"]
            weight = edge.get("abs_partial_corr", edge.get("abs_weight", edge.get("weight", 1.0)))
        else:
            a, b = edge[0], edge[1]
            weight = 1.0
        G.add_edge(a, b, weight=weight)
    
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42)
    
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        font_size=8,
        alpha=0.85,
        node_color='lightblue',
        edge_color='gray',
        arrows=has_directed
    )
    
    plt.title(f"{title} ({len(edges)} total edges, showing {len(show_edges)})")
    plt.tight_layout()
    plt.show()
    
    return G


def visualize_dag(edges, title="Causal DAG", top_k=50):
    """
    Visualize directed acyclic graph (for NOTEARS).
    
    Args:
        edges: List of edge dicts with 'from', 'to', 'abs_weight'
        title: Plot title
        top_k: Maximum edges to display
    
    Returns:
        nx.DiGraph: NetworkX directed graph object
    """
    if not edges:
        print("No edges to visualize")
        return None
    
    # Sort by weight and limit to top_k
    if len(edges) > top_k:
        try:
            sorted_edges = sorted(edges, key=lambda x: x.get("abs_weight", 0), reverse=True)
            show_edges = sorted_edges[:top_k]
        except:
            show_edges = edges[:top_k]
    else:
        show_edges = edges
    
    G = nx.DiGraph()
    
    for edge in show_edges:
        if isinstance(edge, dict):
            a, b = edge["from"], edge["to"]
            weight = edge.get("abs_weight", 1.0)
        else:
            a, b = edge[0], edge[1]
            weight = 1.0
        G.add_edge(a, b, weight=weight)
    
    plt.figure(figsize=(16, 12))
    
    # Try hierarchical layout, fallback to spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)
    
    # Draw edges with varying thickness
    edges_list = list(G.edges(data=True))
    if edges_list:
        weights = [d['weight'] for u, v, d in edges_list]
        max_weight = max(weights) if weights else 1
        widths = [max(0.5, 3 * w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=widths, edge_color='blue', 
                               arrows=True, arrowsize=20, alpha=0.7)
    
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"{title} ({len(edges)} total edges, showing {len(show_edges)})")
    plt.tight_layout()
    plt.show()
    
    # Check for cycles
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print(f"⚠️  WARNING: Found {len(cycles)} cycles in graph!")
        else:
            print("✓ Graph is acyclic (DAG)")
    except:
        pass
    
    return G

# COMMAND ----------

# ===========================
# STATISTICS
# ===========================

def compute_baseline_stats(df):
    """
    Compute baseline statistics for each metric.
    
    Args:
        df: DataFrame with metrics as columns
    
    Returns:
        dict: Statistics per metric (mean, std, median, IQR, etc.)
    """
    baseline = {}
    
    for col in df.columns:
        values = df[col].dropna()
        
        if len(values) == 0:
            baseline[col] = {
                'n': 0, 'mean': None, 'std': None, 'median': None,
                'q1': None, 'q3': None, 'IQR': None, 'min': None, 'max': None
            }
            continue
        
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        baseline[col] = {
            'n': int(len(values)),
            'mean': float(values.mean()),
            'std': float(values.std()),
            'median': float(values.median()),
            'q1': float(q1),
            'q3': float(q3),
            'IQR': float(iqr),
            'min': float(values.min()),
            'max': float(values.max())
        }
    
    return baseline
