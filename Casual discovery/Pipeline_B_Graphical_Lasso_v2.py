# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline B: Graphical Lasso-Based Causal Discovery
# MAGIC
# MAGIC This notebook implements a Graphical Lasso approach for discovering undirected relationships.
# MAGIC - Uses GraphicalLassoCV for automatic alpha selection
# MAGIC - Estimates precision matrix and converts to partial correlations
# MAGIC - Produces undirected graphs (cross-sectional data)
# MAGIC - Uses 65 days of training data

# COMMAND ----------

# MAGIC %pip install networkx scipy scikit-learn

# COMMAND ----------

# MAGIC %run ./causal_discovery_utils

# COMMAND ----------

# Imports
from datetime import datetime
import json
import numpy as np
import pandas as pd
from collections import defaultdict

from pyspark.sql import functions as F

from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

# COMMAND ----------

# ===========================
# PIPELINE-SPECIFIC METHODS
# ===========================

def run_graphical_lasso(df, use_cv=True, cv_alphas=None, manual_alpha=0.1, pcor_threshold=0.1, max_iter=1000):
    """
    Run GraphicalLasso to estimate precision matrix and extract edges.
    
    Args:
        df: Preprocessed DataFrame
        use_cv: Whether to use cross-validation for alpha selection
        cv_alphas: Alpha values for CV search
        manual_alpha: Manual alpha if not using CV
        pcor_threshold: Minimum partial correlation to include edge
        max_iter: Maximum iterations for optimization
    
    Returns:
        dict: Result containing edges, precision matrix, and metadata
    """
    print(f'Running Graphical Lasso (n_samples={len(df)}, n_features={len(df.columns)}, use_cv={use_cv})')
    
    if df.isna().any().any():
        print("❌ Input contains NaN values")
        return {'method': 'graphical-lasso-error', 'error': 'Input contains NaN values', 'edges': None}
    
    if len(df) < 5:
        print(f"❌ Insufficient samples: {len(df)} < 5")
        return {'method': 'graphical-lasso-error', 'error': f'Insufficient samples: {len(df)} < 5', 'edges': None}
    
    n_samples, n_features = df.shape
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
        
        # Get precision matrix and convert to partial correlations
        precision = model.precision_
        d = np.sqrt(np.diag(precision))
        pcor = -precision / np.outer(d, d)
        np.fill_diagonal(pcor, 0.0)
        
        # Extract edges
        edges = []
        edge_weights = []
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
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
        print(f"❌ Graphical Lasso failed: {str(e)}")
        return {
            'method': 'graphical-lasso-error',
            'error': str(e),
            'edges': None,
            'use_cv': use_cv,
            'alpha': manual_alpha,
            'n_samples': n_samples,
            'n_features': n_features
        }


def apply_human_priors_glasso(skeleton_edges, blacklist, whitelist, graphical_lasso_result=None, data=None):
    """
    Apply human priors to GraphicalLasso edges.
    
    Args:
        skeleton_edges: List of edge dicts from GraphicalLasso
        blacklist: Set of (from, to) tuples to remove
        whitelist: Set of (from, to) tuples to add if missing
        graphical_lasso_result: Full result with precision matrix for recovery
        data: DataFrame for correlation estimation
    
    Returns:
        tuple: (kept edges list, review dict)
    """
    kept = []
    removed = []
    
    # Build edge pairs set
    edge_pairs = set()
    for edge in skeleton_edges:
        if isinstance(edge, dict):
            edge_pairs.add((edge["from"], edge["to"]))
            edge_pairs.add((edge["to"], edge["from"]))
        else:
            edge_pairs.add((edge[0], edge[1]))
            edge_pairs.add((edge[1], edge[0]))
    
    available_cols = set(data.columns) if data is not None else set()
    
    # Get partial correlation matrix for weight recovery
    pcor_matrix = graphical_lasso_result.get('partial_correlations') if graphical_lasso_result else None
    columns = list(data.columns) if data is not None else []
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    MIN_PCOR_WEIGHT = 1e-6
    
    # Remove blacklisted edges
    for edge in skeleton_edges:
        if isinstance(edge, dict):
            a, b = edge["from"], edge["to"]
        else:
            a, b = edge[0], edge[1]
        
        if (a, b) in blacklist or (b, a) in blacklist:
            removed.append(edge)
            print(f"  Removed blacklisted edge: {a} -- {b}")
        else:
            if isinstance(edge, dict):
                edge['source'] = 'graphical_lasso'
                kept.append(edge)
            else:
                kept.append({
                    'from': a, 'to': b,
                    'partial_corr': 0.0, 'abs_partial_corr': 0.0,
                    'type': 'undirected', 'source': 'graphical_lasso'
                })
    
    # Add whitelisted edges
    added = []
    for a, b in whitelist:
        if a in available_cols and b in available_cols:
            if (a, b) not in edge_pairs and (b, a) not in edge_pairs:
                pcor_weight = None
                source = None
                
                # Try to recover from precision matrix
                if pcor_matrix is not None and a in col_to_idx and b in col_to_idx:
                    i, j = col_to_idx[a], col_to_idx[b]
                    raw_pcor = pcor_matrix[i, j]
                    if abs(raw_pcor) > MIN_PCOR_WEIGHT:
                        pcor_weight = float(raw_pcor)
                        source = 'whitelist_recovered'
                        print(f"    Recovered from pcor_matrix: {a} -- {b} (pcor={pcor_weight:.6f})")
                
                # Fallback to correlation estimation
                if pcor_weight is None and data is not None:
                    try:
                        corr = data[[a, b]].corr().iloc[0, 1]
                        pcor_weight = float(corr)
                        source = 'whitelist_estimated'
                        print(f"    Estimated via correlation: {a} -- {b} (weight={pcor_weight:.6f})")
                    except Exception as e:
                        print(f"    Warning: Estimation failed for {a} -- {b}: {e}")
                        pcor_weight = 0.0
                        source = 'whitelist_default'
                
                new_edge = {
                    "from": a, "to": b,
                    "partial_corr": pcor_weight if pcor_weight else 0.0,
                    "abs_partial_corr": abs(pcor_weight) if pcor_weight else 0.0,
                    "type": "undirected",
                    "source": source
                }
                added.append(new_edge)
                kept.append(new_edge)
    
    n_recovered = sum(1 for x in added if x.get('source') == 'whitelist_recovered')
    n_estimated = sum(1 for x in added if x.get('source') == 'whitelist_estimated')
    print(f"\nWhitelist Summary:")
    print(f"  - Edges already present: {len([e for e in whitelist if e[0] in available_cols and e[1] in available_cols]) - len(added)}")
    print(f"  - Edges recovered from pcor_matrix: {n_recovered}")
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

# ===========================
# CONFIGURATION
# ===========================

PIPELINE_NAME = "Graphical_Lasso_Based"
METRICS_TABLE = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"
MAX_RUNS_TO_PIVOT = 65

# GraphicalLasso parameters
USE_CV = True
CV_ALPHAS = np.logspace(-3, 1, 20)
MANUAL_ALPHA = 0.1
PCOR_THRESHOLD = 0.1
MAX_ITER = 1000

# Feature selection parameters
TARGET_FEATURES = 45
CORRELATION_THRESHOLD = 0.95

# Artifact path
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"
pipeline_path = f"{path}/{PIPELINE_NAME}"

print(f"Pipeline B Configuration:")
print(f"  - Method: Graphical Lasso-Based")
print(f"  - Training Days: {MAX_RUNS_TO_PIVOT}")
print(f"  - Use CV: {USE_CV}")
print(f"  - Partial Correlation Threshold: {PCOR_THRESHOLD}")
print(f"  - Target Features: {TARGET_FEATURES}")
print(f"  - Artifact Path: {pipeline_path}")

# COMMAND ----------

# ===========================
# DRIVER CODE
# ===========================

print("="*80)
print("PIPELINE B: GRAPHICAL LASSO-BASED CAUSAL DISCOVERY")
print("="*80)

# Step 1: Load metrics data
print("\n[Step 1] Loading metrics data...")
metrics_sdf = spark.sql(f"select date, metric_name, metric_value from {METRICS_TABLE} where date between '2025-10-19' and '2026-02-06'")

metrics_pdf = spark_metrics_to_matrix(metrics_sdf)

# COMMAND ----------

# Step 2: Preprocess
print("\n[Step 2] Preprocessing metrics matrix...")
scaled, preprocess_meta = preprocess_metrics_matrix(
    metrics_pdf,
    zscore=True,
    feature_sample_ratio=2.0
)
print(f"After preprocessing: {scaled.shape}")

# Step 3: Feature Selection
print("\n[Step 3] Feature selection...")
final_features, feature_selection_log = sophisticated_feature_selection(
    scaled,
    target_features=TARGET_FEATURES,
    variance_threshold=1e-6,
    correlation_threshold=CORRELATION_THRESHOLD
)

# Validation
n_samples, n_features = final_features.shape
print(f"\nGraphicalLasso Requirements Check:")
print(f"  Samples: {n_samples}, Features: {n_features}")
print(f"  Sample-to-feature ratio: {n_samples / n_features:.2f}")

if final_features.isna().any().any():
    raise Exception("NaN values detected after feature selection")
if n_features == 0:
    raise Exception("No features remaining after selection")

# COMMAND ----------


# Step 4: Generate blacklist
print("\n[Step 4] Generating blacklist...")
metric_cols = metrics_sdf.select("metric_name").distinct().rdd.flatMap(lambda x: x).collect()
HUMAN_PRIOR_BLACKLIST = generate_stage_blacklist(metric_cols)
print(f"  - Whitelist edges: {len(HUMAN_PRIOR_WHITELIST)}")
print(f"  - Blacklist edges: {len(HUMAN_PRIOR_BLACKLIST)}")

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

if graphical_lasso_result['method'] == 'graphical-lasso-error':
    raise Exception(f"Pipeline B Failed: {graphical_lasso_result['error']}")

# Step 6: Extract raw edges
print(f"\n[Step 6] Extracting edges...")
raw_graphical_lasso_edges = graphical_lasso_result.get('edges', [])
print(f"GraphicalLasso discovered {len(raw_graphical_lasso_edges)} edges")

if len(raw_graphical_lasso_edges) == 0:
    raise Exception("GraphicalLasso found no edges")

# Step 7: Apply blacklist
print(f"\n[Step 7] Filtering blacklist...")
selected_feature_set = set(final_features.columns)
filtered_blacklist = [
    (a, b) for a, b in HUMAN_PRIOR_BLACKLIST 
    if a in selected_feature_set and b in selected_feature_set
]
blacklist_set = set(filtered_blacklist)
print(f"Applicable blacklist edges: {len(filtered_blacklist)}")

# Step 8: Apply human priors
print(f"\n[Step 8] Applying human priors...")
filtered_edges, review = apply_human_priors_glasso(
    raw_graphical_lasso_edges,
    blacklist=blacklist_set,
    whitelist=HUMAN_PRIOR_WHITELIST,
    graphical_lasso_result=graphical_lasso_result,
    data=final_features
)

print(f"After human priors:")
print(f"  - Edges kept: {len(filtered_edges)}")
print(f"  - Edges removed: {len(review.get('removed', []))}")
print(f"  - Edges added: {len(review.get('added', []))}")

# Step 9: Visualize
print(f"\n[Step 9] Visualizing graphs...")
G_raw = visualize_skeleton(raw_graphical_lasso_edges, title="Pipeline B: GraphicalLasso RAW Graph")
G = visualize_skeleton(filtered_edges, title="Pipeline B: GraphicalLasso Graph (After Blacklist/Whitelist)")

print("="*80)
print("PIPELINE B: SUCCESSFUL COMPLETION")
print("="*80)

# COMMAND ----------

# ===========================
# EXPORT ARTIFACTS
# ===========================

print("\n[Step 10] Computing baseline statistics...")
baseline_stats = compute_baseline_stats(final_features)
print(f"Computed baseline statistics for {len(baseline_stats)} metrics")

print("\n[Step 11] Building adjacency maps...")
upstream_map, downstream_map = build_adjacency_maps(filtered_edges, handle_undirected=True)
print(f"Neighbor map: {len(upstream_map)} nodes")

print("\n[Step 12] Exporting artifacts...")
dbutils.fs.mkdirs(pipeline_path)

# Main artifacts
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
    "raw_graphical_lasso_edges": raw_graphical_lasso_edges,
    "filtered_edges": filtered_edges,
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

# Save core artifacts
final_features.to_csv(f"{pipeline_path}/causal_metrics_matrix.csv")
dbutils.fs.put(f"{pipeline_path}/causal_artifacts.json", json.dumps(artifacts, indent=2, default=str), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/baseline_stats.json", json.dumps(baseline_stats, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/upstream_map.json", json.dumps(upstream_map, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/downstream_map.json", json.dumps(downstream_map, indent=2), overwrite=True)

# Save raw graph
raw_rows = []
for edge in raw_graphical_lasso_edges:
    if isinstance(edge, dict):
        raw_rows.append({
            "from": edge["from"], "to": edge["to"],
            "partial_corr": edge.get("partial_corr", 0.0),
            "abs_partial_corr": edge.get("abs_partial_corr", 0.0),
            "edge_type": "undirected", "source": "graphical_lasso_raw"
        })

raw_edges_df = pd.DataFrame(raw_rows)
if 'abs_partial_corr' in raw_edges_df.columns:
    raw_edges_df = raw_edges_df.sort_values("abs_partial_corr", ascending=False)
raw_edges_df.to_csv(f"{pipeline_path}/graphical_lasso_raw_edges.csv", index=False)

raw_upstream_map, raw_downstream_map = build_adjacency_maps(raw_graphical_lasso_edges, handle_undirected=True)
dbutils.fs.put(f"{pipeline_path}/raw_upstream_map.json", json.dumps(raw_upstream_map, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/raw_downstream_map.json", json.dumps(raw_downstream_map, indent=2), overwrite=True)

# Save filtered graph
rows = []
for edge in filtered_edges:
    if isinstance(edge, dict):
        rows.append({
            "from": edge["from"], "to": edge["to"],
            "partial_corr": edge.get("partial_corr", 0.0),
            "abs_partial_corr": edge.get("abs_partial_corr", 0.0),
            "edge_type": "undirected",
            "source": edge.get("source", "graphical_lasso")
        })

cand_df = pd.DataFrame(rows)
if 'abs_partial_corr' in cand_df.columns:
    cand_df = cand_df.sort_values("abs_partial_corr", ascending=False)
cand_df.to_csv(f"{pipeline_path}/graphical_lasso_causal_edges.csv", index=False)

print("="*80)
print("✓ PIPELINE B COMPLETE — All artifacts saved")
print("="*80)
print(f"\nSaved to: {pipeline_path}")
print(f"\nFinal Results:")
print(f"  - GraphicalLasso alpha: {graphical_lasso_result['alpha']:.4f}")
print(f"  - RAW edges: {len(raw_graphical_lasso_edges)}")
print(f"  - FILTERED edges: {len(filtered_edges)}")
print(f"  - Removed: {len(review.get('removed', []))}")
print(f"  - Added: {len(review.get('added', []))}")