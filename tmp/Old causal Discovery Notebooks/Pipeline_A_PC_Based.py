# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline A: PC-Based Causal Discovery
# MAGIC
# MAGIC This notebook implements a pure PC algorithm approach for causal discovery.
# MAGIC - Uses PC algorithm with configurable alpha
# MAGIC - No fallback to other methods (logs failures)
# MAGIC - Strict human priors application
# MAGIC - Uses 65 days of training data

# COMMAND ----------

# MAGIC %pip install networkx scipy scikit-learn pydot

# COMMAND ----------

pip install statsmodels

# COMMAND ----------

pip install causal-learn

# COMMAND ----------

# Imports
from datetime import datetime
import json
import numpy as np
import numpy.linalg as la
import pandas as pd
from collections import defaultdict

from pyspark.sql import functions as F
from pyspark.sql import types as T

# Causality libs
from causallearn.search.ConstraintBased.PC import pc

# ML Libs
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Visualization
import networkx as nx
import matplotlib.pyplot as plt

# COMMAND ----------

# Configuration for Pipeline A
PIPELINE_NAME = "PC_Based"
METRICS_TABLE = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"
MAX_RUNS_TO_PIVOT = 65  # Using 65 days instead of 45
PC_ALPHA = 0.05  # Configurable PC alpha
PC_INDEP_TEST = 'fisherz'  # Independence test method

# DBFS path for artifacts
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"
pipeline_path = f"{path}/{PIPELINE_NAME}"

print(f"Pipeline A Configuration:")
print(f"  - Method: PC Algorithm Only")
print(f"  - Training Days: {MAX_RUNS_TO_PIVOT}")
print(f"  - PC Alpha: {PC_ALPHA}")
print(f"  - Independence Test: {PC_INDEP_TEST}")
print(f"  - Artifact Path: {pipeline_path}")

# COMMAND ----------

def generate_stage_blacklist(metric_cols):
    """Generate conservative blacklist pairs from metric column names.

    Rules encoded:
    - forbid any edge from `silver_` metrics to `bronze_` or `raw_` metrics
    - forbid any edge from `bronze_` metrics to `raw_` metrics
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

# COMMAND ----------

def preprocess_metrics_matrix(
    df: pd.DataFrame,
    *,
    zscore: bool = True,
    impute_strategy: str = "median",
    max_missing_frac: float = 0.5,
    feature_sample_ratio: float = 2.5,   # More aggressive for 65 days
    min_keep_features: int = 20,
):
    """Preprocess metrics matrix for causal discovery."""
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
    
    # Add missingness indicators for partially missing columns
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

def sophisticated_feature_selection_for_pc(
    df: pd.DataFrame,
    *,
    target_features: int = 40,  # Target 30-45 features
    variance_threshold: float = 1e-6,
    correlation_threshold: float = 0.9,
):
    """Sophisticated feature selection optimized for PC algorithm stability."""
    print(f"Starting sophisticated feature selection (target: {target_features} features)")
    print(f"Input features: {df.shape[1]}")
    
    selection_log = {
        "initial_features": df.shape[1],
        "target_features": target_features,
        "steps": []
    }
    
    out = df.copy()
    
    # Step 1: Variance Filter (keep existing logic)
    print("\n[Feature Selection Step 1] Variance filtering...")
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
    
    # Step 2: Redundant Metric Removal (by naming patterns)
    print("\n[Feature Selection Step 2] Redundant metric removal...")
    redundant_removed = []
    
    # Group metrics by type and remove redundant ones
    current_cols = list(out.columns)
    
    # Remove multiple percentiles (keep mean + p95 only)
    percentile_groups = {}
    for col in current_cols:
        if any(p in col for p in ['_p25', '_p50', '_p75', '_p90', '_p99']):
            base_name = col.split('_p')[0]
            if base_name not in percentile_groups:
                percentile_groups[base_name] = []
            percentile_groups[base_name].append(col)
    
    for base_name, percentile_cols in percentile_groups.items():
        # Keep only p95 if available, otherwise keep the highest percentile
        if any('_p95' in col for col in percentile_cols):
            keep_col = next(col for col in percentile_cols if '_p95' in col)
        else:
            keep_col = percentile_cols[-1]  # Keep highest percentile
        
        to_remove = [col for col in percentile_cols if col != keep_col]
        redundant_removed.extend(to_remove)
    
    # Remove duplicate row counts from same pipeline step
    row_count_groups = {}
    for col in current_cols:
        if any(term in col.lower() for term in ['rows', 'count', 'records']):
            # Group by pipeline stage
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
        if len(count_cols) > 2:  # Keep at most 2 count metrics per stage
            # Prefer input/output rows over intermediate counts
            priority_patterns = ['input_rows', 'output_rows', 'record_count']
            prioritized = []
            others = []
            
            for col in count_cols:
                if any(pattern in col for pattern in priority_patterns):
                    prioritized.append(col)
                else:
                    others.append(col)
            
            keep_cols = prioritized[:2] if len(prioritized) >= 2 else prioritized + others[:2-len(prioritized)]
            to_remove = [col for col in count_cols if col not in keep_cols]
            redundant_removed.extend(to_remove)
    
    # Apply redundant removal
    if redundant_removed:
        out = out.drop(columns=redundant_removed)
        print(f"  Removed {len(redundant_removed)} redundant features")
        selection_log["steps"].append({
            "step": "redundant_removal",
            "removed_count": len(redundant_removed),
            "removed_features": redundant_removed,
            "reason": "structural_redundancy"
        })
    
    # Step 3: High-Correlation Pruning with Smart Preferences
    print("\n[Feature Selection Step 3] Correlation-based pruning...")
    if out.shape[1] > target_features:
        corr_matrix = out.corr().abs()
        
        # Find highly correlated pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
        
        # Smart removal preferences
        def get_feature_priority(feature_name):
            """Higher score = higher priority (keep feature)."""
            score = 0
            
            # Prefer rates over raw counts
            if any(term in feature_name.lower() for term in ['rate', 'ratio', 'percent', 'pct']):
                score += 10
            
            # Prefer aggregated KPIs over raw metrics
            if any(term in feature_name.lower() for term in ['mean', 'avg', 'median', 'p95']):
                score += 5
            
            # Prefer downstream (silver) over upstream
            if feature_name.startswith('silver_'):
                score += 8
            elif feature_name.startswith('bronze_'):
                score += 4
            elif feature_name.startswith('raw_'):
                score += 0
            
            # Prefer business metrics over technical metrics
            if any(term in feature_name.lower() for term in ['fuel', 'speed', 'distance', 'trip']):
                score += 6
            
            # Prefer duration metrics (often important for pipeline analysis)
            if 'duration' in feature_name.lower():
                score += 3
            
            return score
        
        # Remove features from correlated pairs
        correlation_removed = []
        remaining_cols = set(out.columns)
        
        for col1, col2, corr_val in sorted(corr_pairs, key=lambda x: x[2], reverse=True):
            if col1 in remaining_cols and col2 in remaining_cols:
                # Remove the lower priority feature
                priority1 = get_feature_priority(col1)
                priority2 = get_feature_priority(col2)
                
                if priority1 >= priority2:
                    remove_col = col2
                    keep_col = col1
                else:
                    remove_col = col1
                    keep_col = col2
                
                correlation_removed.append(remove_col)
                remaining_cols.remove(remove_col)
                print(f"    Removed {remove_col} (corr={corr_val:.3f} with {keep_col})")
        
        if correlation_removed:
            out = out.drop(columns=correlation_removed)
            selection_log["steps"].append({
                "step": "correlation_pruning",
                "removed_count": len(correlation_removed),
                "removed_features": correlation_removed,
                "reason": f"correlation > {correlation_threshold}",
                "correlation_threshold": correlation_threshold
            })
    
    # Step 4: Final size adjustment (if still too many features)
    print("\n[Feature Selection Step 4] Final size adjustment...")
    if out.shape[1] > target_features:
        # Use variance-based ranking for final selection
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
    selection_log["reduction_ratio"] = selection_log["total_removed"] / df.shape[1]
    
    print(f"\n✓ Feature selection complete:")
    print(f"  Initial: {df.shape[1]} features")
    print(f"  Final: {out.shape[1]} features")
    print(f"  Removed: {selection_log['total_removed']} features ({selection_log['reduction_ratio']:.1%})")
    print(f"  Target achieved: {'✓' if out.shape[1] <= target_features else '✗'}")
    
    return out, selection_log

# COMMAND ----------

def extract_pc_edges(pc_result, column_names):
    """Extract directed edges from PC algorithm result."""
    edges = []
    
    if not hasattr(pc_result, 'G') or pc_result.G is None:
        return edges
    
    # Try to get adjacency matrix from PC result
    try:
        if hasattr(pc_result.G, 'graph'):
            graph_matrix = pc_result.G.graph
        elif hasattr(pc_result.G, 'adj_matrix'):
            graph_matrix = pc_result.G.adj_matrix
        else:
            # Try direct access
            graph_matrix = pc_result.G
    except:
        print("Could not extract graph matrix from PC result")
        return edges
    
    if graph_matrix is None or graph_matrix.shape[0] != len(column_names):
        return edges
    
    for i in range(len(column_names)):
        for j in range(len(column_names)):
            if i != j and graph_matrix[i, j] != 0:
                edge_value = graph_matrix[i, j]
                
                # PC edge encoding: 1=directed (i->j), -1=directed (j->i), 2=undirected
                if edge_value == 1:
                    edges.append((column_names[i], column_names[j], 'directed'))
                elif edge_value == -1:
                    edges.append((column_names[j], column_names[i], 'directed'))
                elif edge_value == 2:
                    # For undirected, add both directions
                    edges.append((column_names[i], column_names[j], 'undirected'))
                    edges.append((column_names[j], column_names[i], 'undirected'))
    
    return edges

# COMMAND ----------

def run_pc_algorithm(df: pd.DataFrame, alpha=0.05, indep_test='fisherz'):
    """Run PC algorithm with detailed logging and no fallback."""
    print(f'Running PC Algorithm (n_samples={len(df)}, n_features={len(df.columns)}, alpha={alpha}, test={indep_test})')
    
    # Validate input
    if df.isna().any().any():
        return {'method': 'pc-error', 'error': 'Input contains NaN values', 'edges': None}
    
    if len(df) < 10:
        return {'method': 'pc-error', 'error': f'Insufficient samples: {len(df)} < 10', 'edges': None}
    
    # Check sample to feature ratio
    n_samples, n_features = df.shape
    ratio = n_samples / n_features
    print(f"Sample-to-feature ratio: {ratio:.2f}")
    
    if ratio < 2.0:
        print(f"⚠️  WARNING: Low sample-to-feature ratio ({ratio:.2f}). PC may be unreliable.")
    
    try:
        data = df.values.astype(float)
        
        # Run PC with specified parameters
        pc_obj = pc(data, alpha=alpha, indep_test=indep_test)
        
        # Extract edges
        edges = extract_pc_edges(pc_obj, df.columns.tolist())
        
        print(f"✓ PC Algorithm successful: found {len(edges)} edges")
        
        return {
            'method': 'pc-success',
            'pc_object': pc_obj,
            'edges': edges,
            'alpha': alpha,
            'indep_test': indep_test,
            'n_samples': n_samples,
            'n_features': n_features,
            'sample_feature_ratio': ratio
        }
        
    except Exception as e:
        error_msg = f"PC Algorithm failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            'method': 'pc-error',
            'error': error_msg,
            'edges': None,
            'alpha': alpha,
            'indep_test': indep_test,
            'n_samples': n_samples,
            'n_features': n_features
        }

# COMMAND ----------

def estimate_weight_via_regression(from_col, to_col, data):
    """Estimate causal weight using OLS regression: to_col ~ from_col.
    
    Returns the regression coefficient as the edge weight.
    This provides a data-driven weight for whitelisted edges.
    """
    try:
        X = data[from_col].values.reshape(-1, 1)
        y = data[to_col].values
        
        # Simple OLS: y = beta * X + intercept
        # beta = (X'X)^-1 X'y
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        
        return float(beta[1])  # Return coefficient (not intercept)
    except Exception as e:
        print(f"    Warning: Regression failed for {from_col} -> {to_col}: {e}")
        return 0.0


def apply_human_priors(skeleton_edges, blacklist, whitelist, pc_result=None, data=None):
    """Apply human priors to skeleton edges with data-driven weights.
    
    HYBRID APPROACH for whitelist edges:
    - Edges already discovered: keep original edge
    - Whitelist edges not present: estimate weight via OLS regression
    
    Args:
        skeleton_edges: List of (from, to, edge_type) tuples from PC
        blacklist: Set of (from, to) tuples to remove
        whitelist: Set of (from, to) tuples to add if missing
        pc_result: PC algorithm result (for extracting p-values if available)
        data: DataFrame used for causal discovery (for OLS estimation)
    """
    kept = []
    removed = []
    present = set((a, b) for a, b, _ in skeleton_edges)
    
    # Get available columns from data
    available_cols = set(data.columns) if data is not None else set()
    
    # Remove blacklisted edges
    for a, b, edge_type in skeleton_edges:
        if (a, b) in blacklist:
            removed.append({'from': a, 'to': b, 'edge_type': edge_type, 'reason': 'blacklisted'})
            print(f"  Removed blacklisted edge: {a} -> {b}")
        else:
            kept.append({
                'from': a,
                'to': b,
                'edge_type': edge_type,
                'weight': 1.0,  # PC doesn't provide weights, use 1.0
                'source': 'pc_algorithm'
            })
    
    # Add whitelisted edges if missing (with data-driven weights)
    added = []
    for a, b in whitelist:
        if a in available_cols and b in available_cols:
            if (a, b) not in present:
                # Edge not discovered by PC - estimate weight via OLS
                weight = estimate_weight_via_regression(a, b, data) if data is not None else 0.0
                source = 'whitelist_estimated'
                print(f"    Estimated via OLS: {a} -> {b} (weight={weight:.6f})")
                
                edge_dict = {
                    'from': a,
                    'to': b,
                    'edge_type': 'directed',
                    'weight': weight,
                    'abs_weight': abs(weight),
                    'source': source
                }
                added.append(edge_dict)
                kept.append(edge_dict)
    
    # Summary
    n_estimated = sum(1 for x in added if x.get('source') == 'whitelist_estimated')
    print(f"\nWhitelist Summary:")
    print(f"  - Edges already present (kept): {len([e for e in whitelist if e[0] in available_cols and e[1] in available_cols]) - len(added)}")
    print(f"  - Edges estimated via OLS: {n_estimated}")
    print(f"  - Total whitelist edges added: {len(added)}")
    
    review = {
        'removed': removed, 
        'added': added,
        'whitelist_estimated_via_ols': n_estimated
    }
    return kept, review

# COMMAND ----------

def visualize_skeleton(edges, title="Causal Skeleton", top_k=50):
    """Visualize causal skeleton."""
    if not edges:
        print("No edges to visualize")
        return None
    
    # Show top_k edges
    show_edges = edges[:top_k] if len(edges) > top_k else edges
    
    G = nx.DiGraph()
    for a, b, edge_type in show_edges:
        G.add_edge(a, b, edge_type=edge_type)
    
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
        edge_color='gray'
    )
    
    plt.title(f"{title} ({len(edges)} total edges)")
    plt.tight_layout()
    plt.show()
    
    return G

# COMMAND ----------

def spark_metrics_to_matrix(spark, metrics_table, max_runs=65, date_col='date'):
    """Read metrics table and pivot to wide pandas DataFrame."""
    sdf = spark.table(metrics_table).select(
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

# Define Human Priors
HUMAN_PRIOR_WHITELIST = [
    ("raw_input_record_count", "bronze_input_rows"),
    ("raw_ingestion_duration_sec", "bronze_ingestion_duration_sec"),
    ("raw_min_trip_start_ts", "bronze_input_rows"),
    ("raw_max_trip_end_ts", "bronze_output_rows"),
    ("raw_distance_mean", "bronze_distance_km_mean"),
    ("raw_avg_speed_mean", "silver_avg_speed_imputed"),
    ("raw_unique_units", "bronze_output_rows"),
    ("raw_null_count_unit_id", "bronze_null_primary_key_rows"),
    ("bronze_input_rows", "silver_input_data_count"),
    ("bronze_output_rows", "silver_count_after_feature_engineering"),
    ("bronze_survival_rate", "silver_survival_rate"),
    ("bronze_distance_km_mean", "silver_avg_speed_imputed"),
    ("bronze_duration_mean", "silver_avg_speed_imputed"),
    ("bronze_ingestion_duration_sec", "silver_ingestion_duration_sec"),
    ("bronze_distance_km_mean", "mean_fuel_per_100km"),
    ("bronze_distance_km_mean", "p50_fuel_per_100km"),
    ("bronze_distance_km_mean", "p95_fuel_per_100km"),
    ("silver_ml_imputed_fuel_mean", "silver_ml_prediction_mean"),
    ("silver_ml_residual_mean", "silver_ml_abs_residual_mean"),
]

# Get metric columns for blacklist generation
metric_cols = (
    spark.table(METRICS_TABLE)
    .select("metric_name")
    .distinct()
    .rdd.flatMap(lambda x: x)
    .collect()
)
HUMAN_PRIOR_BLACKLIST = generate_stage_blacklist(metric_cols)

print(f"Human Priors Configuration:")
print(f"  - Whitelist edges: {len(HUMAN_PRIOR_WHITELIST)}")
print(f"  - Blacklist edges: {len(HUMAN_PRIOR_BLACKLIST)}")

# COMMAND ----------

# ===========================
# MAIN PIPELINE EXECUTION
# ===========================

print("="*80)
print("PIPELINE A: PC-BASED CAUSAL DISCOVERY")
print("="*80)

# Step 1: Load metrics data
print("\n[Step 1] Loading metrics data...")
metrics_pdf = spark_metrics_to_matrix(
    spark, 
    metrics_table=METRICS_TABLE,
    max_runs=MAX_RUNS_TO_PIVOT
)

# Step 2: Preprocess metrics matrix
print("\n[Step 2] Preprocessing metrics matrix...")
scaled, preprocess_meta = preprocess_metrics_matrix(
    metrics_pdf,
    zscore=True,
    feature_sample_ratio=2.5
)
print(f"After preprocessing: {scaled.shape}")

# Step 3: Sophisticated Feature Selection for PC Stability
print("\n[Step 3] Sophisticated feature selection for PC algorithm...")
final_features, feature_selection_log = sophisticated_feature_selection_for_pc(
    scaled,
    target_features=40,  # Target 30-45 features for PC stability
    variance_threshold=1e-6,
    correlation_threshold=0.95
)

# Check PC requirements
n_samples, n_features = final_features.shape
sample_to_feature_ratio = n_samples / n_features
print(f"\nPC Algorithm Requirements Check:")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  Sample-to-feature ratio: {sample_to_feature_ratio:.2f}")
print(f"  PC stability: {'✓ Good' if sample_to_feature_ratio > 1.5 else '⚠ Marginal' if sample_to_feature_ratio > 1.0 else '✗ Poor'}")

# Step 4: Final validation
print("\n[Step 4] Final validation...")
if final_features.isna().any().any():
    print("⚠️  WARNING: NaN values detected after feature selection")
    
if n_features == 0:
    print("🛑 PIPELINE A TERMINATED - No features remaining after selection")
    raise Exception("No features remaining after feature selection")

print(f"✓ Ready for PC algorithm with {n_features} carefully selected features")

display(final_features)


# COMMAND ----------


# Step 5: Run PC Algorithm (NO FALLBACK)
print(f"\n[Step 5] Running PC Algorithm (alpha={PC_ALPHA}, test={PC_INDEP_TEST})...")
pc_result = run_pc_algorithm(
    final_features, 
    alpha=PC_ALPHA, 
    indep_test=PC_INDEP_TEST
)

print(f"PC Result: {pc_result['method']}")
if pc_result['method'] == 'pc-error':
    print(f"PC Failed: {pc_result['error']}")
    print("🛑 PIPELINE A TERMINATED - No fallback method")
    raise Exception(f"Pipeline A Failed: {pc_result['error']}")

# COMMAND ----------

# Step 6: Extract edges from PC result (RAW - before human priors)
print(f"\n[Step 6] Extracting edges from PC result...")
raw_pc_edges = pc_result.get('edges', [])
print(f"PC discovered {len(raw_pc_edges)} directed edges (before filtering)")

if len(raw_pc_edges) == 0:
    print("🛑 PIPELINE A TERMINATED - PC found no edges")
    raise Exception("Pipeline A: PC algorithm found no edges")

# Step 7: Apply blacklist filtering as POST-PROCESSING
print(f"\n[Step 7] Applying blacklist filtering (post-processing)...")

# Generate blacklist for selected features only
selected_feature_set = set(final_features.columns)
filtered_blacklist = [
    (a, b) for a, b in HUMAN_PRIOR_BLACKLIST 
    if a in selected_feature_set and b in selected_feature_set
]
blacklist_set = set(filtered_blacklist)
print(f"Applicable blacklist edges: {len(filtered_blacklist)}")

# Step 8: Apply human priors (blacklist + whitelist with data-driven weights)
print(f"\n[Step 8] Applying human priors...")
filtered_edges, review = apply_human_priors(
    raw_pc_edges,
    blacklist=blacklist_set,
    whitelist=HUMAN_PRIOR_WHITELIST,
    pc_result=pc_result,
    data=final_features
)

print(f"After human priors:")
print(f"  - Edges kept: {len(filtered_edges)}")
print(f"  - Edges removed: {len(review.get('removed', []))}")
print(f"  - Edges added: {len(review.get('added', []))}")

# Step 9: Visualize RAW DAG (before human priors)
print(f"\n[Step 9] Visualizing RAW causal graph (before blacklist)...")
G_raw = visualize_skeleton(raw_pc_edges, title="Pipeline A: PC RAW Graph (Before Blacklist)")

# Step 10: Visualize FILTERED DAG (after human priors)
print(f"\n[Step 10] Visualizing FILTERED causal graph...")
pc_edges = raw_pc_edges  # For backward compatibility
# Convert filtered_edges to tuple format for visualization
filtered_edges_tuples = [(e['from'], e['to'], e.get('edge_type', 'directed')) for e in filtered_edges]
G = visualize_skeleton(filtered_edges_tuples, title="Pipeline A: PC Graph (After Blacklist/Whitelist)")

print("="*80)
print("PIPELINE A: SUCCESSFUL COMPLETION")
print("="*80)

# COMMAND ----------

# ===========================
# EXPORT ARTIFACTS
# ===========================

print("\n[Step 11] Computing baseline statistics...")

def compute_baseline_stats(df):
    """Compute baseline statistics for each metric."""
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

baseline_stats = compute_baseline_stats(final_features)
print(f"Computed baseline statistics for {len(baseline_stats)} metrics")

# COMMAND ----------

print("\n[Step 12] Building adjacency maps...")

def build_adjacency_maps(edges):
    """Build upstream and downstream adjacency lists from directed edge list.
    
    For PC directed edges:
    - upstream_map[node] = list of nodes that have edges TO this node (parents)
    - downstream_map[node] = list of nodes that this node has edges TO (children)
    """
    upstream_map = defaultdict(list)
    downstream_map = defaultdict(list)
    
    for edge in edges:
        if isinstance(edge, dict):
            parent, child = edge["from"], edge["to"]
        else:
            parent, child = edge[0], edge[1]
        
        # Directed edges: parent -> child
        upstream_map[child].append(parent)    # child's upstream (parents)
        downstream_map[parent].append(child)  # parent's downstream (children)
    
    return dict(upstream_map), dict(downstream_map)

upstream_map, downstream_map = build_adjacency_maps(filtered_edges)

print(f"Upstream map: {len(upstream_map)} nodes have parents")
print(f"Downstream map: {len(downstream_map)} nodes have children")

# COMMAND ----------

print("\n[Step 13] Exporting all artifacts...")

# Create pipeline directory
dbutils.fs.mkdirs(pipeline_path)

# Main artifacts dictionary
artifacts = {
    "pipeline": PIPELINE_NAME,
    "method": "pc-based",
    "data_type": "cross-sectional",
    "status": "SUCCESS",
    "training_days": MAX_RUNS_TO_PIVOT,
    "preprocess_meta": preprocess_meta,
    "feature_selection_log": feature_selection_log,
    "pc_result": {
        "method": pc_result["method"],
        "alpha": pc_result["alpha"],
        "indep_test": pc_result["indep_test"],
        "n_samples": pc_result["n_samples"],
        "n_features": pc_result["n_features"],
        "sample_feature_ratio": pc_result["sample_feature_ratio"],
        "raw_edges_found": len(raw_pc_edges)
    },
    "edge_stats": {
        "raw_edges": len(raw_pc_edges),
        "removed_by_blacklist": len(review.get('removed', [])),
        "added_from_whitelist": len(review.get('added', [])),
        "final_edges": len(filtered_edges),
        "edge_type": "directed",
        "orientation_method": "pc_algorithm"
    },
    "raw_pc_edges": [(e[0], e[1], e[2]) for e in raw_pc_edges],  # PC output before human priors
    "filtered_edges": filtered_edges,  # Final edges after human priors
    "blacklist_filtering": {
        "blacklist_edges_applicable": len(filtered_blacklist),
        "edges_removed": [(e['from'], e['to']) for e in review.get('removed', [])],
        "whitelist_edges_added": len(review.get('added', [])),
        "whitelist_estimated_via_ols": review.get('whitelist_estimated_via_ols', 0),
        "whitelist_details": [(e['from'], e['to'], e.get('source', ''), e.get('weight', 0)) for e in review.get('added', [])]
    },
    "final_graph_stats": {
        "total_edges": len(filtered_edges),
        "nodes_with_parents": len(upstream_map),
        "nodes_with_children": len(downstream_map)
    }
}

# Save main artifacts
final_features.to_csv(f"{pipeline_path}/causal_metrics_matrix.csv")
dbutils.fs.put(f"{pipeline_path}/causal_artifacts.json", 
               json.dumps(artifacts, indent=2, default=str), 
               overwrite=True)

# Save baseline statistics
dbutils.fs.put(f"{pipeline_path}/baseline_stats.json", 
               json.dumps(baseline_stats, indent=2), 
               overwrite=True)

# Save adjacency maps
dbutils.fs.put(f"{pipeline_path}/upstream_map.json", 
               json.dumps(upstream_map, indent=2), 
               overwrite=True)

dbutils.fs.put(f"{pipeline_path}/downstream_map.json", 
               json.dumps(downstream_map, indent=2), 
               overwrite=True)

# ===========================
# SAVE RAW PC GRAPH (before human priors)
# ===========================
print("\nSaving RAW PC graph (before human priors)...")

# Save raw edges as CSV
raw_rows = []
for edge in raw_pc_edges:
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        raw_rows.append({
            "from": edge[0],
            "to": edge[1],
            "edge_type": edge[2] if len(edge) > 2 else 'directed',
            "weight": 1.0,  # PC doesn't provide weights
            "source": "pc_raw"
        })

raw_edges_df = pd.DataFrame(raw_rows)
raw_edges_df.to_csv(f"{pipeline_path}/pc_raw_edges.csv", index=False)

# Build adjacency maps for RAW graph
raw_upstream_map = defaultdict(list)
raw_downstream_map = defaultdict(list)
for edge in raw_pc_edges:
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        parent, child = edge[0], edge[1]
        raw_upstream_map[child].append(parent)
        raw_downstream_map[parent].append(child)

dbutils.fs.put(f"{pipeline_path}/raw_upstream_map.json", 
               json.dumps(dict(raw_upstream_map), indent=2), 
               overwrite=True)

dbutils.fs.put(f"{pipeline_path}/raw_downstream_map.json", 
               json.dumps(dict(raw_downstream_map), indent=2), 
               overwrite=True)

print(f"  - Raw graph: {len(raw_pc_edges)} edges")
print(f"  - Raw upstream map: {len(raw_upstream_map)} nodes")
print(f"  - Raw downstream map: {len(raw_downstream_map)} nodes")

# ===========================
# SAVE FILTERED GRAPH (after human priors)
# ===========================
print("\nSaving FILTERED graph (after human priors)...")

# Generate PC edge list for filtered edges
rows = []
for edge in filtered_edges:
    if isinstance(edge, dict):
        rows.append({
            "from": edge["from"],
            "to": edge["to"],
            "edge_type": edge.get("edge_type", "directed"),
            "weight": edge.get("weight", 1.0),
            "abs_weight": abs(edge.get("weight", 1.0)),
            "source": edge.get("source", "pc_algorithm")
        })
    elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
        rows.append({
            "from": edge[0],
            "to": edge[1],
            "edge_type": edge[2] if len(edge) > 2 else 'directed',
            "weight": 1.0,
            "abs_weight": 1.0,
            "source": "pc_algorithm"
        })

cand_df = pd.DataFrame(rows)
if 'abs_weight' in cand_df.columns:
    cand_df = cand_df.sort_values("abs_weight", ascending=False)
cand_df.to_csv(f"{pipeline_path}/pc_causal_edges.csv", index=False)

print(f"  - Filtered graph: {len(filtered_edges)} edges")
print(f"  - Filtered upstream map: {len(upstream_map)} nodes")
print(f"  - Filtered downstream map: {len(downstream_map)} nodes")

print("="*80)
print("✓ PIPELINE A COMPLETE — All artifacts saved")
print("="*80)
print(f"\nSaved artifacts to: {pipeline_path}")
print(f"\n📁 ARTIFACT INVENTORY:")
print(f"\n  [Core Artifacts]")
print(f"  - causal_artifacts.json          → Main pipeline metadata & both edge lists")
print(f"  - causal_metrics_matrix.csv      → Feature matrix used for discovery")
print(f"  - baseline_stats.json            → Statistical baselines for each metric")
print(f"\n  [RAW PC Graph - Before Human Priors]")
print(f"  - pc_raw_edges.csv               → {len(raw_pc_edges)} edges from pure PC")
print(f"  - raw_upstream_map.json          → Parent nodes for each node (raw)")
print(f"  - raw_downstream_map.json        → Child nodes for each node (raw)")
print(f"\n  [FILTERED Graph - After Human Priors]")
print(f"  - pc_causal_edges.csv            → {len(filtered_edges)} edges after blacklist/whitelist")
print(f"  - upstream_map.json              → Parent nodes for each node (filtered)")
print(f"  - downstream_map.json            → Child nodes for each node (filtered)")

print(f"\nFinal Results:")
print(f"  - Training period: {MAX_RUNS_TO_PIVOT} days (cross-sectional)")
print(f"  - Final feature matrix: {final_features.shape}")
print(f"  - RAW graph edges (PC output): {len(raw_pc_edges)}")
print(f"  - Removed by blacklist: {len(review.get('removed', []))}")
print(f"  - Added from whitelist: {len(review.get('added', []))}")
print(f"  - FILTERED graph edges (after priors): {len(filtered_edges)}")
print(f"  - Nodes with parents: {len(upstream_map)}")
print(f"  - Nodes with children: {len(downstream_map)}")
print(f"  - Edge type: Directed")