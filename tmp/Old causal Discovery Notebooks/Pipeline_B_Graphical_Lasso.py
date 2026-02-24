# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline B: Graphical Lasso-Based Causal Discovery
# MAGIC 
# MAGIC This notebook implements a Graphical Lasso approach for discovering undirected relationships.
# MAGIC - Uses GraphicalLassoCV for automatic alpha selection
# MAGIC - Estimates precision matrix and converts to partial correlations
# MAGIC - Thresholds weak edges with configurable threshold
# MAGIC - Produces undirected graphs (no temporal orientation - data is cross-sectional)
# MAGIC - Uses sophisticated feature selection targeting 45 features for stability

# COMMAND ----------

# MAGIC %pip install networkx scipy scikit-learn

# COMMAND ----------

# No additional package installations needed for GraphicalLasso
# scikit-learn is sufficient for GraphicalLasso functionality

# COMMAND ----------

# Imports
from datetime import datetime
import json
import numpy as np
import pandas as pd
from collections import defaultdict

from pyspark.sql import functions as F
from pyspark.sql import types as T

# ML Libs
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

# Remove statsmodels import since Granger tests are no longer needed

# Visualization
import networkx as nx
import matplotlib.pyplot as plt

# COMMAND ----------

# Configuration for Pipeline B
PIPELINE_NAME = "Graphical_Lasso_Based"
METRICS_TABLE = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"
MAX_RUNS_TO_PIVOT = 65  # Using 65 days instead of 45

# Graphical Lasso parameters
USE_CV = True  # Use cross-validation for alpha selection
CV_ALPHAS = np.logspace(-3, 1, 20)  # Alpha range for CV
MANUAL_ALPHA = 0.1  # Manual alpha if not using CV
PCOR_THRESHOLD = 0.1  # Threshold for partial correlation edges
MAX_ITER = 1000  # Maximum iterations

# DBFS path for artifacts
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"
pipeline_path = f"{path}/{PIPELINE_NAME}"

print(f"Pipeline B Configuration:")
print(f"  - Method: Graphical Lasso-Based")
print(f"  - Training Days: {MAX_RUNS_TO_PIVOT}")
print(f"  - Use CV: {USE_CV}")
print(f"  - Alpha Range: {CV_ALPHAS[0]:.3f} - {CV_ALPHAS[-1]:.3f}" if USE_CV else f"  - Manual Alpha: {MANUAL_ALPHA}")
print(f"  - Partial Correlation Threshold: {PCOR_THRESHOLD}")
print(f"  - Artifact Path: {pipeline_path}")

# COMMAND ----------

def generate_stage_blacklist(metric_cols):
    """Generate conservative blacklist pairs from metric column names."""
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
    feature_sample_ratio: float = 2.0,   # More conservative for GraphicalLasso
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
    
    # Z-score scaling (ESSENTIAL for GraphicalLasso)
    if zscore:
        scaler = StandardScaler()
        imputed_array = scaler.fit_transform(imputed_array)
    
    df_clean = pd.DataFrame(
        imputed_array,
        index=df_num.index,
        columns=df_num.columns,
    )
    
    # Feature reduction if features >> samples (more conservative for GraphicalLasso)
    n_samples, n_feats = df_clean.shape
    meta["pre_reduction_shape"] = (n_samples, n_feats)
    
    if n_feats > n_samples * feature_sample_ratio:
        var = df_clean.var().sort_values(ascending=False)
        keep_k = max(int(n_samples * 1.2), min_keep_features)
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

def sophisticated_feature_selection_for_graphical_lasso(
    df: pd.DataFrame,
    *,
    target_features: int = 45,  # Slightly higher for GraphicalLasso tolerance
    variance_threshold: float = 1e-6,
    correlation_threshold: float = 0.95,  # Lower threshold for GraphicalLasso
):
    """Sophisticated feature selection optimized for GraphicalLasso stability."""
    print(f"Starting sophisticated feature selection (target: {target_features} features)")
    print(f"Input features: {df.shape[1]}")
    
    selection_log = {
        "initial_features": df.shape[1],
        "target_features": target_features,
        "steps": []
    }
    
    out = df.copy()
    
    # Step 1: Variance Filter
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
    
    # Step 2: Redundant Metric Removal
    print("\n[Feature Selection Step 2] Redundant metric removal...")
    redundant_removed = []
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
        if any('_p95' in col for col in percentile_cols):
            keep_col = next(col for col in percentile_cols if '_p95' in col)
        else:
            keep_col = percentile_cols[-1]
        
        to_remove = [col for col in percentile_cols if col != keep_col]
        redundant_removed.extend(to_remove)
    
    # Remove duplicate row counts from same pipeline step
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
    
    if redundant_removed:
        out = out.drop(columns=redundant_removed)
        print(f"  Removed {len(redundant_removed)} redundant features")
        selection_log["steps"].append({
            "step": "redundant_removal",
            "removed_count": len(redundant_removed),
            "removed_features": redundant_removed,
            "reason": "structural_redundancy"
        })
    
    # Step 3: High-Correlation Pruning (more aggressive for GraphicalLasso)
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
        
        def get_feature_priority(feature_name):
            """Higher score = higher priority (keep feature)."""
            score = 0
            
            if any(term in feature_name.lower() for term in ['rate', 'ratio', 'percent', 'pct']):
                score += 10
            
            if any(term in feature_name.lower() for term in ['mean', 'avg', 'median', 'p95']):
                score += 5
            
            if feature_name.startswith('silver_'):
                score += 8
            elif feature_name.startswith('bronze_'):
                score += 4
            elif feature_name.startswith('raw_'):
                score += 0
            
            if any(term in feature_name.lower() for term in ['fuel', 'speed', 'distance', 'trip']):
                score += 6
            
            if 'duration' in feature_name.lower():
                score += 3
            
            return score
        
        # Remove features from correlated pairs
        correlation_removed = []
        remaining_cols = set(out.columns)
        
        for col1, col2, corr_val in sorted(corr_pairs, key=lambda x: x[2], reverse=True):
            if col1 in remaining_cols and col2 in remaining_cols:
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
    
    # Step 4: Final size adjustment
    print("\n[Feature Selection Step 4] Final size adjustment...")
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
    selection_log["reduction_ratio"] = selection_log["total_removed"] / df.shape[1]
    
    print(f"\n✓ Feature selection complete:")
    print(f"  Initial: {df.shape[1]} features")
    print(f"  Final: {out.shape[1]} features")
    print(f"  Removed: {selection_log['total_removed']} features ({selection_log['reduction_ratio']:.1%})")
    print(f"  Target achieved: {'✓' if out.shape[1] <= target_features else '✗'}")
    
    return out, selection_log

# COMMAND ----------

def run_graphical_lasso(df: pd.DataFrame, use_cv=True, cv_alphas=None, manual_alpha=0.1, pcor_threshold=0.1, max_iter=1000):
    """Run GraphicalLasso to estimate precision matrix and extract edges."""
    print(f'Running Graphical Lasso (n_samples={len(df)}, n_features={len(df.columns)}, use_cv={use_cv})')
    
    # Validate input
    if df.isna().any().any():
        return {'method': 'graphical-lasso-error', 'error': 'Input contains NaN values', 'edges': None}
    
    if len(df) < 5:
        return {'method': 'graphical-lasso-error', 'error': f'Insufficient samples: {len(df)} < 5', 'edges': None}
    
    n_samples, n_features = df.shape
    
    # Check sample to feature ratio
    ratio = n_samples / n_features
    print(f"Sample-to-feature ratio: {ratio:.2f}")
    
    if ratio < 1.5:
        print(f"⚠️  WARNING: Very low sample-to-feature ratio ({ratio:.2f}). Results may be unstable.")
    
    X = df.values
    cols = df.columns.tolist()
    
    try:
        if use_cv and cv_alphas is not None:
            print(f"Using cross-validation with {len(cv_alphas)} alpha values")
            model = GraphicalLassoCV(alphas=cv_alphas, max_iter=max_iter, cv=3)
            model.fit(X)
            selected_alpha = model.alpha_
            print(f"Selected alpha via CV: {selected_alpha:.4f}")
        else:
            selected_alpha = manual_alpha
            print(f"Using manual alpha: {selected_alpha}")
            model = GraphicalLasso(alpha=selected_alpha, max_iter=max_iter)
            model.fit(X)
        
        # Get precision matrix
        precision = model.precision_
        
        # Convert precision to partial correlations
        d = np.sqrt(np.diag(precision))
        pcor = -precision / np.outer(d, d)
        np.fill_diagonal(pcor, 0.0)
        
        # Extract edges based on partial correlation threshold
        edges = []
        edge_weights = []
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):  # Only upper triangle (undirected)
                val = pcor[i, j]
                if abs(val) >= pcor_threshold:
                    edges.append({
                        "from": cols[i],
                        "to": cols[j],
                        "partial_corr": float(val),
                        "abs_partial_corr": float(abs(val)),
                        "type": "undirected"
                    })
                    edge_weights.append(abs(val))
        
        # Sort edges by strength
        edges = sorted(edges, key=lambda x: x["abs_partial_corr"], reverse=True)
        
        print(f"✓ Graphical Lasso successful: found {len(edges)} edges")
        
        return {
            'method': 'graphical-lasso-success',
            'model': model,
            'edges': edges,
            'precision_matrix': precision,
            'partial_correlations': pcor,
            'alpha': selected_alpha,
            'use_cv': use_cv,
            'pcor_threshold': pcor_threshold,
            'n_samples': n_samples,
            'n_features': n_features,
            'sample_feature_ratio': ratio,
            'edge_weights_stats': {
                'min': float(np.min(edge_weights)) if edge_weights else 0,
                'max': float(np.max(edge_weights)) if edge_weights else 0,
                'mean': float(np.mean(edge_weights)) if edge_weights else 0,
                'std': float(np.std(edge_weights)) if edge_weights else 0
            }
        }
        
    except Exception as e:
        error_msg = f"Graphical Lasso failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            'method': 'graphical-lasso-error',
            'error': error_msg,
            'edges': None,
            'use_cv': use_cv,
            'alpha': manual_alpha,
            'n_samples': n_samples,
            'n_features': n_features
        }

# COMMAND ----------

# Note: GraphicalLasso produces undirected edges based on partial correlations
# No edge orientation is needed since data represents independent daily snapshots

# COMMAND ----------

def apply_human_priors(skeleton_edges, blacklist, whitelist, graphical_lasso_result=None, data=None):
    """Apply human priors to skeleton edges with data-driven weights.
    
    HYBRID APPROACH for whitelist edges:
    - Edges already discovered: keep original partial correlation
    - Whitelist edges below threshold: recover from full precision matrix
    - Whitelist edges truly absent: estimate via sample partial correlation
    
    Args:
        skeleton_edges: List of edge dictionaries from GraphicalLasso
        blacklist: Set of (from, to) tuples to remove
        whitelist: Set of (from, to) tuples to add if missing
        graphical_lasso_result: Full result including precision matrix for weight recovery
        data: DataFrame used for causal discovery (for partial correlation estimation)
    """
    kept = []
    removed = []
    
    # Convert edges to (from, to) format for comparison
    edge_pairs = set()
    for edge in skeleton_edges:
        if isinstance(edge, dict):
            edge_pairs.add((edge["from"], edge["to"]))
            edge_pairs.add((edge["to"], edge["from"]))  # Undirected - both directions
        else:
            edge_pairs.add((edge[0], edge[1]))
            edge_pairs.add((edge[1], edge[0]))
    
    # Get available columns from data
    available_cols = set(data.columns) if data is not None else set()
    
    # Get precision matrix and partial correlations for weight recovery
    precision_matrix = graphical_lasso_result.get('precision_matrix') if graphical_lasso_result else None
    pcor_matrix = graphical_lasso_result.get('partial_correlations') if graphical_lasso_result else None
    columns = list(data.columns) if data is not None else []
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    
    # Minimum threshold for considering recovered weight as "present"
    MIN_PCOR_WEIGHT = 1e-6
    
    # Remove blacklisted edges
    for edge in skeleton_edges:
        if isinstance(edge, dict):
            a, b = edge["from"], edge["to"]
        else:
            a, b = edge[0], edge[1]
        
        if (a, b) in blacklist or (b, a) in blacklist:  # Check both directions for undirected
            removed.append(edge)
            print(f"  Removed blacklisted edge: {a} -- {b}")
        else:
            # Keep existing edge with its partial correlation weight
            if isinstance(edge, dict):
                edge['source'] = 'graphical_lasso'
                kept.append(edge)
            else:
                kept.append({
                    'from': a,
                    'to': b,
                    'partial_corr': 0.0,
                    'abs_partial_corr': 0.0,
                    'type': 'undirected',
                    'source': 'graphical_lasso'
                })
    
    # Add whitelisted edges if missing (with data-driven weights)
    added = []
    for a, b in whitelist:
        if a in available_cols and b in available_cols:
            if (a, b) not in edge_pairs and (b, a) not in edge_pairs:
                # Edge not in final graph - need to add it with appropriate weight
                
                pcor_weight = None
                source = None
                
                # Option 1: Try to recover partial correlation from full precision matrix
                if pcor_matrix is not None and a in col_to_idx and b in col_to_idx:
                    i = col_to_idx[a]
                    j = col_to_idx[b]
                    raw_pcor = pcor_matrix[i, j]
                    
                    if abs(raw_pcor) > MIN_PCOR_WEIGHT:
                        # Partial correlation exists but was below threshold
                        pcor_weight = float(raw_pcor)
                        source = 'whitelist_recovered'
                        print(f"    Recovered from pcor_matrix: {a} -- {b} (pcor={pcor_weight:.6f})")
                
                # Option 2: Estimate partial correlation from data if not found
                if pcor_weight is None and data is not None:
                    # Estimate partial correlation using sample covariance
                    try:
                        # Simple approach: use correlation as approximation
                        # (full partial correlation would need inverting residual covariance)
                        corr = data[[a, b]].corr().iloc[0, 1]
                        pcor_weight = float(corr)
                        source = 'whitelist_estimated'
                        print(f"    Estimated via correlation: {a} -- {b} (weight={pcor_weight:.6f})")
                    except Exception as e:
                        print(f"    Warning: Estimation failed for {a} -- {b}: {e}")
                        pcor_weight = 0.0
                        source = 'whitelist_default'
                
                new_edge = {
                    "from": a,
                    "to": b,
                    "partial_corr": pcor_weight if pcor_weight else 0.0,
                    "abs_partial_corr": abs(pcor_weight) if pcor_weight else 0.0,
                    "type": "undirected",
                    "source": source
                }
                added.append(new_edge)
                kept.append(new_edge)
    
    # Summary
    n_recovered = sum(1 for x in added if x.get('source') == 'whitelist_recovered')
    n_estimated = sum(1 for x in added if x.get('source') == 'whitelist_estimated')
    print(f"\nWhitelist Summary:")
    print(f"  - Edges already present (kept): {len([e for e in whitelist if e[0] in available_cols and e[1] in available_cols]) - len(added)}")
    print(f"  - Edges recovered from pcor_matrix (below threshold): {n_recovered}")
    print(f"  - Edges estimated via correlation: {n_estimated}")
    print(f"  - Total whitelist edges added: {len(added)}")
    
    review = {
        'removed': removed, 
        'added': added,
        'whitelist_recovered_from_pcor': n_recovered,
        'whitelist_estimated_via_corr': n_estimated
    }
    return kept, review

# COMMAND ----------

def visualize_skeleton(edges, title="Causal Skeleton", top_k=50):
    """Visualize causal skeleton with edge types."""
    if not edges:
        print("No edges to visualize")
        return None
    
    # Show top_k strongest edges
    if len(edges) > top_k:
        # Sort by strength if available
        try:
            sorted_edges = sorted(edges, key=lambda x: x.get("abs_partial_corr", 0), reverse=True)
            show_edges = sorted_edges[:top_k]
        except:
            show_edges = edges[:top_k]
    else:
        show_edges = edges
    
    G = nx.Graph()  # Start with undirected graph
    directed_edges = []
    
    for edge in show_edges:
        if isinstance(edge, dict):
            a, b = edge["from"], edge["to"]
            edge_type = edge.get("type", "undirected")
            
            if edge_type == "directed":
                directed_edges.append((a, b))
                G.add_edge(a, b, weight=edge.get("abs_partial_corr", 1.0), edge_type=edge_type)
            else:
                G.add_edge(a, b, weight=edge.get("abs_partial_corr", 1.0), edge_type=edge_type)
        else:
            a, b = edge[0], edge[1]
            G.add_edge(a, b, weight=1.0, edge_type="undirected")
    
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    
    # Draw undirected edges
    undirected_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") != "directed"]
    if undirected_edges:
        nx.draw_networkx_edges(G, pos, edgelist=undirected_edges, edge_color='gray', alpha=0.6)
    
    # Draw directed edges
    if directed_edges:
        nx.draw_networkx_edges(G, pos, edgelist=directed_edges, edge_color='blue', 
                             arrows=True, arrowsize=20, alpha=0.8)
    
    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"{title} ({len(edges)} total edges, {len(directed_edges)} directed)")
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

# Define Human Priors (same as Pipeline A)
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
print("PIPELINE B: GRAPHICAL LASSO-BASED CAUSAL DISCOVERY")
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
    feature_sample_ratio=2.0  # More conservative for GraphicalLasso
)
print(f"After preprocessing: {scaled.shape}")

# Step 3: Sophisticated Feature Selection for GraphicalLasso
print("\n[Step 3] Sophisticated feature selection for GraphicalLasso...")
final_features, feature_selection_log = sophisticated_feature_selection_for_graphical_lasso(
    scaled,
    target_features=45,  # Slightly higher target for GraphicalLasso
    variance_threshold=1e-6,
    correlation_threshold=0.95
)

# Check GraphicalLasso requirements
n_samples, n_features = final_features.shape
sample_to_feature_ratio = n_samples / n_features
print(f"\nGraphicalLasso Requirements Check:")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  Sample-to-feature ratio: {sample_to_feature_ratio:.2f}")
print(f"  GraphicalLasso stability: {'✓ Good' if sample_to_feature_ratio > 1.3 else '⚠ Marginal' if sample_to_feature_ratio > 1.0 else '✗ Poor'}")

# Step 4: Final validation
print("\n[Step 4] Final validation...")
if final_features.isna().any().any():
    print("⚠️  WARNING: NaN values detected after feature selection")
    
if n_features == 0:
    print("🛑 PIPELINE B TERMINATED - No features remaining after selection")
    raise Exception("No features remaining after feature selection")

print(f"✓ Ready for GraphicalLasso with {n_features} carefully selected features")

# Step 5: Run GraphicalLasso
print(f"\n[Step 5] Running GraphicalLasso...")
graphical_lasso_result = run_graphical_lasso(
    final_features,
    use_cv=USE_CV,
    cv_alphas=CV_ALPHAS if USE_CV else None,
    manual_alpha=MANUAL_ALPHA,
    pcor_threshold=PCOR_THRESHOLD,
    max_iter=MAX_ITER
)

print(f"Graphical Lasso Result: {graphical_lasso_result['method']}")
if graphical_lasso_result['method'] == 'graphical-lasso-error':
    print(f"Graphical Lasso Failed: {graphical_lasso_result['error']}")
    print("🛑 PIPELINE B TERMINATED")
    raise Exception(f"Pipeline B Failed: {graphical_lasso_result['error']}")

# Step 6: Extract edges from Graphical Lasso result
undirected_edges = graphical_lasso_result.get('edges', [])
print(f"Graphical Lasso discovered {len(undirected_edges)} undirected edges")

if len(undirected_edges) == 0:
    print("🛑 PIPELINE B TERMINATED - Graphical Lasso found no edges")
    raise Exception("Pipeline B: Graphical Lasso found no edges")

# Step 7: Keep GraphicalLasso edges as undirected (appropriate for partial correlations)
print(f"\n[Step 7] Keeping edges undirected (GraphicalLasso partial correlations)...")
print(f"GraphicalLasso edges are inherently undirected: {len(undirected_edges)} edges")
print(f"  - All edges represent conditional independence/partial correlation")
print(f"  - No temporal orientation applied (data is cross-sectional)")

# Save raw edges for later comparison
raw_graphical_lasso_edges = undirected_edges.copy()

# Step 8: Apply blacklist filtering as POST-PROCESSING
print(f"\n[Step 8] Applying blacklist filtering (post-processing)...")

# Generate blacklist for selected features only
selected_feature_set = set(final_features.columns)
filtered_blacklist = [
    (a, b) for a, b in HUMAN_PRIOR_BLACKLIST 
    if a in selected_feature_set and b in selected_feature_set
]
blacklist_set = set(filtered_blacklist)
print(f"Applicable blacklist edges: {len(filtered_blacklist)}")

# Step 9: Apply human priors (blacklist + whitelist with data-driven weights)
print(f"\n[Step 9] Applying human priors...")
filtered_edges, review = apply_human_priors(
    undirected_edges,
    blacklist=blacklist_set,
    whitelist=HUMAN_PRIOR_WHITELIST,
    graphical_lasso_result=graphical_lasso_result,
    data=final_features
)

print(f"After human priors:")
print(f"  - Edges kept: {len(filtered_edges)}")
print(f"  - Edges removed: {len(review.get('removed', []))}")
print(f"  - Edges added: {len(review.get('added', []))}")

# Step 10: Visualize RAW graph (before human priors)
print(f"\n[Step 10] Visualizing RAW causal skeleton (before blacklist)...")
G_raw = visualize_skeleton(raw_graphical_lasso_edges, title="Pipeline B: GraphicalLasso RAW Graph (Before Blacklist)")

# Step 11: Visualize FILTERED graph (after human priors)
print(f"\n[Step 11] Visualizing FILTERED causal skeleton...")
G = visualize_skeleton(filtered_edges, title="Pipeline B: GraphicalLasso Graph (After Blacklist/Whitelist)")

print("="*80)
print("PIPELINE B: SUCCESSFUL COMPLETION")
print("="*80)

# COMMAND ----------

# ===========================
# EXPORT ARTIFACTS
# ===========================

print("\n[Step 12] Computing baseline statistics...")

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

print("\n[Step 13] Building adjacency maps...")

def build_adjacency_maps(edges):
    """Build upstream and downstream adjacency lists from edge list."""
    upstream_map = defaultdict(list)
    downstream_map = defaultdict(list)
    
    for edge in edges:
        if isinstance(edge, dict):
            parent, child = edge["from"], edge["to"]
        else:
            parent, child = edge[0], edge[1]
        
        # For undirected edges, add both directions
        edge_type = edge.get("type", "undirected") if isinstance(edge, dict) else "undirected"
        
        if edge_type == "directed":
            # Directed: only one direction
            upstream_map[child].append(parent)
            downstream_map[parent].append(child)
        else:
            # Undirected: both directions
            upstream_map[child].append(parent)
            downstream_map[parent].append(child)
            upstream_map[parent].append(child)
            downstream_map[child].append(parent)
    
    return dict(upstream_map), dict(downstream_map)

upstream_map, downstream_map = build_adjacency_maps(filtered_edges)

print(f"Upstream map: {len(upstream_map)} nodes have parents")
print(f"Downstream map: {len(downstream_map)} nodes have children")

# COMMAND ----------

print("\n[Step 14] GraphicalLasso edges are inherently undirected...")
print("GraphicalLasso provides undirected partial correlation relationships")
print("No temporal causality testing - data represents independent daily snapshots")

# COMMAND ----------

print("\n[Step 15] Exporting all artifacts...")

# Create pipeline directory
dbutils.fs.mkdirs(pipeline_path)

# Main artifacts dictionary
artifacts = {
    "pipeline": PIPELINE_NAME,
    "method": "graphical-lasso-based",
    "data_type": "cross-sectional",
    "status": "SUCCESS",
    "training_days": MAX_RUNS_TO_PIVOT,
    "preprocess_meta": preprocess_meta,
    "feature_selection_log": feature_selection_log,
    "graphical_lasso_result": {
        "method": graphical_lasso_result["method"],
        "alpha": graphical_lasso_result["alpha"],
        "use_cv": graphical_lasso_result["use_cv"],
        "pcor_threshold": graphical_lasso_result["pcor_threshold"],
        "n_samples": graphical_lasso_result["n_samples"],
        "n_features": graphical_lasso_result["n_features"],
        "sample_feature_ratio": graphical_lasso_result["sample_feature_ratio"],
        "raw_edges_found": len(raw_graphical_lasso_edges),
        "edge_weights_stats": graphical_lasso_result.get("edge_weights_stats", {})
    },
    "edge_stats": {
        "raw_edges": len(raw_graphical_lasso_edges),
        "removed_by_blacklist": len(review.get('removed', [])),
        "added_from_whitelist": len(review.get('added', [])),
        "final_edges": len(filtered_edges),
        "edge_type": "undirected",
        "weight_type": "partial_correlation"
    },
    "raw_graphical_lasso_edges": raw_graphical_lasso_edges,  # GraphicalLasso output before human priors
    "filtered_edges": filtered_edges,  # Final edges after human priors
    "blacklist_filtering": {
        "blacklist_edges_applicable": len(filtered_blacklist),
        "edges_removed": [(e['from'], e['to']) for e in review.get('removed', []) if isinstance(e, dict)],
        "whitelist_edges_added": len(review.get('added', [])),
        "whitelist_recovered_from_pcor": review.get('whitelist_recovered_from_pcor', 0),
        "whitelist_estimated_via_corr": review.get('whitelist_estimated_via_corr', 0),
        "whitelist_details": [(e['from'], e['to'], e.get('source', ''), e.get('partial_corr', 0)) for e in review.get('added', [])]
    },
    "final_graph_stats": {
        "total_edges": len(filtered_edges),
        "nodes_with_neighbors": len(upstream_map)
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

# Save adjacency maps (for undirected graph, upstream=downstream=neighbors)
dbutils.fs.put(f"{pipeline_path}/upstream_map.json", 
               json.dumps(upstream_map, indent=2), 
               overwrite=True)

dbutils.fs.put(f"{pipeline_path}/downstream_map.json", 
               json.dumps(downstream_map, indent=2), 
               overwrite=True)

# ===========================
# SAVE RAW GRAPHICALLASSO GRAPH (before human priors)
# ===========================
print("\nSaving RAW GraphicalLasso graph (before human priors)...")

# Save raw edges as CSV
raw_rows = []
for edge in raw_graphical_lasso_edges:
    if isinstance(edge, dict):
        raw_rows.append({
            "from": edge["from"],
            "to": edge["to"],
            "partial_corr": edge.get("partial_corr", 0.0),
            "abs_partial_corr": edge.get("abs_partial_corr", 0.0),
            "edge_type": "undirected",
            "source": "graphical_lasso_raw"
        })

raw_edges_df = pd.DataFrame(raw_rows)
if 'abs_partial_corr' in raw_edges_df.columns:
    raw_edges_df = raw_edges_df.sort_values("abs_partial_corr", ascending=False)
raw_edges_df.to_csv(f"{pipeline_path}/graphical_lasso_raw_edges.csv", index=False)

# Build adjacency maps for RAW graph
raw_upstream_map = defaultdict(list)
raw_downstream_map = defaultdict(list)
for edge in raw_graphical_lasso_edges:
    if isinstance(edge, dict):
        a, b = edge["from"], edge["to"]
        # Undirected: add both directions
        raw_upstream_map[b].append(a)
        raw_downstream_map[a].append(b)
        raw_upstream_map[a].append(b)
        raw_downstream_map[b].append(a)

dbutils.fs.put(f"{pipeline_path}/raw_upstream_map.json", 
               json.dumps(dict(raw_upstream_map), indent=2), 
               overwrite=True)

dbutils.fs.put(f"{pipeline_path}/raw_downstream_map.json", 
               json.dumps(dict(raw_downstream_map), indent=2), 
               overwrite=True)

print(f"  - Raw graph: {len(raw_graphical_lasso_edges)} edges")
print(f"  - Raw neighbor map: {len(raw_upstream_map)} nodes")

# ===========================
# SAVE FILTERED GRAPH (after human priors)
# ===========================
print("\nSaving FILTERED graph (after human priors)...")

# Generate GraphicalLasso edge list ranked by partial correlation strength
rows = []
for edge in filtered_edges:
    if isinstance(edge, dict):
        rows.append({
            "from": edge["from"],
            "to": edge["to"],
            "partial_corr": edge.get("partial_corr", 0.0),
            "abs_partial_corr": edge.get("abs_partial_corr", 0.0),
            "edge_type": "undirected",
            "source": edge.get("source", "graphical_lasso")
        })

cand_df = pd.DataFrame(rows)
if 'abs_partial_corr' in cand_df.columns:
    cand_df = cand_df.sort_values("abs_partial_corr", ascending=False)
cand_df.to_csv(f"{pipeline_path}/graphical_lasso_causal_edges.csv", index=False)

print(f"  - Filtered graph: {len(filtered_edges)} edges")
print(f"  - Filtered neighbor map: {len(upstream_map)} nodes")

print("="*80)
print("✓ PIPELINE B COMPLETE — All artifacts saved")
print("="*80)
print(f"\nSaved artifacts to: {pipeline_path}")
print(f"\n📁 ARTIFACT INVENTORY:")
print(f"\n  [Core Artifacts]")
print(f"  - causal_artifacts.json              → Main pipeline metadata & both edge lists")
print(f"  - causal_metrics_matrix.csv          → Feature matrix used for discovery")
print(f"  - baseline_stats.json                → Statistical baselines for each metric")
print(f"\n  [RAW GraphicalLasso Graph - Before Human Priors]")
print(f"  - graphical_lasso_raw_edges.csv      → {len(raw_graphical_lasso_edges)} edges from pure GraphicalLasso")
print(f"  - raw_upstream_map.json              → Neighbor nodes for each node (raw)")
print(f"  - raw_downstream_map.json            → Neighbor nodes for each node (raw)")
print(f"\n  [FILTERED Graph - After Human Priors]")
print(f"  - graphical_lasso_causal_edges.csv   → {len(filtered_edges)} edges after blacklist/whitelist")
print(f"  - upstream_map.json                  → Neighbor nodes for each node (filtered)")
print(f"  - downstream_map.json                → Neighbor nodes for each node (filtered)")

print(f"\nFinal Results:")
print(f"  - Training period: {MAX_RUNS_TO_PIVOT} days (cross-sectional)")
print(f"  - Final feature matrix: {final_features.shape}")
print(f"  - GraphicalLasso alpha: {graphical_lasso_result['alpha']:.4f}")
print(f"  - RAW graph edges (GraphicalLasso output): {len(raw_graphical_lasso_edges)}")
print(f"  - Removed by blacklist: {len(review.get('removed', []))}")
print(f"  - Added from whitelist: {len(review.get('added', []))}")
print(f"  - FILTERED graph edges (after priors): {len(filtered_edges)}")
print(f"  - Connected nodes: {len(set([e['from'] for e in filtered_edges] + [e['to'] for e in filtered_edges if isinstance(e, dict)]))}")
print(f"  - Edge type: Undirected partial correlations")