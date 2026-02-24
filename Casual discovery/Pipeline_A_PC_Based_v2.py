# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline A: PC-Based Causal Discovery
# MAGIC
# MAGIC This notebook implements a pure PC algorithm approach for causal discovery.
# MAGIC - Uses PC algorithm with configurable alpha
# MAGIC - No fallback to other methods
# MAGIC - Applies human priors (blacklist + whitelist)
# MAGIC - Uses 65 days of training data (cross-sectional)

# COMMAND ----------

# MAGIC %pip install networkx scipy scikit-learn pydot causal-learn

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

from causallearn.search.ConstraintBased.PC import pc

# COMMAND ----------

# ===========================
# PIPELINE-SPECIFIC METHODS
# ===========================

def extract_pc_edges(pc_result, column_names):
    """
    Extract directed edges from PC algorithm result.
    
    Args:
        pc_result: Result object from causal-learn PC algorithm
        column_names: List of column names corresponding to node indices
    
    Returns:
        list: List of (from, to, edge_type) tuples
    """
    edges = []
    
    if not hasattr(pc_result, 'G') or pc_result.G is None:
        return edges
    
    try:
        if hasattr(pc_result.G, 'graph'):
            graph_matrix = pc_result.G.graph
        elif hasattr(pc_result.G, 'adj_matrix'):
            graph_matrix = pc_result.G.adj_matrix
        else:
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
                    edges.append((column_names[i], column_names[j], 'undirected'))
                    edges.append((column_names[j], column_names[i], 'undirected'))
    
    return edges


def run_pc_algorithm(df, alpha=0.05, indep_test='fisherz'):
    """
    Run PC algorithm on data.
    
    Args:
        df: Preprocessed DataFrame
        alpha: Significance level for independence tests
        indep_test: Independence test method ('fisherz', 'chisq', etc.)
    
    Returns:
        dict: Result containing edges, metadata, and PC object
    """
    print(f'Running PC Algorithm (n_samples={len(df)}, n_features={len(df.columns)}, alpha={alpha})')
    
    if df.isna().any().any():
        print("❌ Input contains NaN values")
        return {'method': 'pc-error', 'error': 'Input contains NaN values', 'edges': None}
    
    if len(df) < 10:
        print(f"❌ Insufficient samples: {len(df)} < 10")
        return {'method': 'pc-error', 'error': f'Insufficient samples: {len(df)} < 10', 'edges': None}
    
    n_samples, n_features = df.shape
    ratio = n_samples / n_features
    print(f"Sample-to-feature ratio: {ratio:.2f}")
    
    if ratio < 2.0:
        print(f"⚠️  WARNING: Low sample-to-feature ratio ({ratio:.2f}). PC may be unreliable.")
    
    try:
        data = df.values.astype(float)
        pc_obj = pc(data, alpha=alpha, indep_test=indep_test)
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
        print(f"❌ PC Algorithm failed: {str(e)}")
        return {
            'method': 'pc-error',
            'error': str(e),
            'edges': None,
            'alpha': alpha,
            'indep_test': indep_test,
            'n_samples': n_samples,
            'n_features': n_features
        }


def estimate_weight_via_regression(from_col, to_col, data):
    """
    Estimate causal weight using OLS regression.
    
    Args:
        from_col: Source column name
        to_col: Target column name
        data: DataFrame with the columns
    
    Returns:
        float: Regression coefficient
    """
    try:
        X = data[from_col].values.reshape(-1, 1)
        y = data[to_col].values
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        return float(beta[1])
    except Exception as e:
        print(f"    Warning: Regression failed for {from_col} -> {to_col}: {e}")
        return 0.0


def apply_human_priors_pc(skeleton_edges, blacklist, whitelist, data=None):
    """
    Apply human priors to PC skeleton edges.
    
    Args:
        skeleton_edges: List of (from, to, edge_type) tuples from PC
        blacklist: Set of (from, to) tuples to remove
        whitelist: Set of (from, to) tuples to add if missing
        data: DataFrame for OLS weight estimation
    
    Returns:
        tuple: (kept edges list, review dict)
    """
    kept = []
    removed = []
    present = set((a, b) for a, b, _ in skeleton_edges)
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
                'weight': 1.0,
                'source': 'pc_algorithm'
            })
    
    # Add whitelisted edges with OLS-estimated weights
    added = []
    for a, b in whitelist:
        if a in available_cols and b in available_cols:
            if (a, b) not in present:
                weight = estimate_weight_via_regression(a, b, data) if data is not None else 0.0
                print(f"    Estimated via OLS: {a} -> {b} (weight={weight:.6f})")
                
                edge_dict = {
                    'from': a,
                    'to': b,
                    'edge_type': 'directed',
                    'weight': weight,
                    'abs_weight': abs(weight),
                    'source': 'whitelist_estimated'
                }
                added.append(edge_dict)
                kept.append(edge_dict)
    
    n_estimated = len(added)
    print(f"\nWhitelist Summary:")
    print(f"  - Edges already present: {len([e for e in whitelist if e[0] in available_cols and e[1] in available_cols]) - n_estimated}")
    print(f"  - Edges estimated via OLS: {n_estimated}")
    print(f"  - Total whitelist edges added: {n_estimated}")
    
    review = {
        'removed': removed, 
        'added': added,
        'whitelist_estimated_via_ols': n_estimated
    }
    return kept, review

# COMMAND ----------

# ===========================
# CONFIGURATION
# ===========================

PIPELINE_NAME = "PC_Based"
METRICS_TABLE = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"
MAX_RUNS_TO_PIVOT = 65

# PC Algorithm parameters
PC_ALPHA = 0.05
PC_INDEP_TEST = 'fisherz'

# Feature selection parameters
TARGET_FEATURES = 40
CORRELATION_THRESHOLD = 0.95

# Artifact path
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"
pipeline_path = f"{path}/{PIPELINE_NAME}"

print(f"Pipeline A Configuration:")
print(f"  - Method: PC Algorithm Only")
print(f"  - Training Days: {MAX_RUNS_TO_PIVOT}")
print(f"  - PC Alpha: {PC_ALPHA}")
print(f"  - Independence Test: {PC_INDEP_TEST}")
print(f"  - Target Features: {TARGET_FEATURES}")
print(f"  - Artifact Path: {pipeline_path}")

# COMMAND ----------

# ===========================
# DRIVER CODE
# ===========================

print("="*80)
print("PIPELINE A: PC-BASED CAUSAL DISCOVERY")
print("="*80)

# Step 1: Load metrics data
print("\n[Step 1] Loading metrics data...")
metrics_sdf = spark.table(METRICS_TABLE)
metrics_pdf = spark_metrics_to_matrix(metrics_sdf, max_runs=MAX_RUNS_TO_PIVOT)

# Step 2: Preprocess
print("\n[Step 2] Preprocessing metrics matrix...")
scaled, preprocess_meta = preprocess_metrics_matrix(
    metrics_pdf,
    zscore=True,
    feature_sample_ratio=2.5
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
print(f"\nPC Requirements Check:")
print(f"  Samples: {n_samples}, Features: {n_features}")
print(f"  Sample-to-feature ratio: {n_samples / n_features:.2f}")

if final_features.isna().any().any():
    raise Exception("NaN values detected after feature selection")
if n_features == 0:
    raise Exception("No features remaining after selection")

# Step 4: Generate blacklist
print("\n[Step 4] Generating blacklist...")
metric_cols = metrics_sdf.select("metric_name").distinct().rdd.flatMap(lambda x: x).collect()
HUMAN_PRIOR_BLACKLIST = generate_stage_blacklist(metric_cols)
print(f"  - Whitelist edges: {len(HUMAN_PRIOR_WHITELIST)}")
print(f"  - Blacklist edges: {len(HUMAN_PRIOR_BLACKLIST)}")

# Step 5: Run PC Algorithm
print(f"\n[Step 5] Running PC Algorithm...")
pc_result = run_pc_algorithm(final_features, alpha=PC_ALPHA, indep_test=PC_INDEP_TEST)

if pc_result['method'] == 'pc-error':
    raise Exception(f"Pipeline A Failed: {pc_result['error']}")

# Step 6: Extract raw edges
print(f"\n[Step 6] Extracting edges...")
raw_pc_edges = pc_result.get('edges', [])
print(f"PC discovered {len(raw_pc_edges)} edges (before filtering)")

if len(raw_pc_edges) == 0:
    raise Exception("PC found no edges")

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
filtered_edges, review = apply_human_priors_pc(
    raw_pc_edges,
    blacklist=blacklist_set,
    whitelist=HUMAN_PRIOR_WHITELIST,
    data=final_features
)

print(f"After human priors:")
print(f"  - Edges kept: {len(filtered_edges)}")
print(f"  - Edges removed: {len(review.get('removed', []))}")
print(f"  - Edges added: {len(review.get('added', []))}")

# Step 9: Visualize
print(f"\n[Step 9] Visualizing graphs...")
G_raw = visualize_skeleton(raw_pc_edges, title="Pipeline A: PC RAW Graph (Before Blacklist)")

filtered_edges_tuples = [(e['from'], e['to'], e.get('edge_type', 'directed')) for e in filtered_edges]
G = visualize_skeleton(filtered_edges_tuples, title="Pipeline A: PC Graph (After Blacklist/Whitelist)")

print("="*80)
print("PIPELINE A: SUCCESSFUL COMPLETION")
print("="*80)

# COMMAND ----------

# ===========================
# EXPORT ARTIFACTS
# ===========================

print("\n[Step 10] Computing baseline statistics...")
baseline_stats = compute_baseline_stats(final_features)
print(f"Computed baseline statistics for {len(baseline_stats)} metrics")

print("\n[Step 11] Building adjacency maps...")
upstream_map, downstream_map = build_adjacency_maps(filtered_edges, handle_undirected=False)
print(f"Upstream map: {len(upstream_map)} nodes")
print(f"Downstream map: {len(downstream_map)} nodes")

print("\n[Step 12] Exporting artifacts...")
dbutils.fs.mkdirs(pipeline_path)

# Main artifacts
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
    "raw_pc_edges": [(e[0], e[1], e[2]) for e in raw_pc_edges],
    "filtered_edges": filtered_edges,
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

# Save core artifacts
final_features.to_csv(f"{pipeline_path}/causal_metrics_matrix.csv")
dbutils.fs.put(f"{pipeline_path}/causal_artifacts.json", json.dumps(artifacts, indent=2, default=str), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/baseline_stats.json", json.dumps(baseline_stats, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/upstream_map.json", json.dumps(upstream_map, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/downstream_map.json", json.dumps(downstream_map, indent=2), overwrite=True)

# Save raw graph
raw_rows = [{"from": e[0], "to": e[1], "edge_type": e[2] if len(e) > 2 else 'directed', "weight": 1.0, "source": "pc_raw"} 
            for e in raw_pc_edges if isinstance(e, (list, tuple)) and len(e) >= 2]
pd.DataFrame(raw_rows).to_csv(f"{pipeline_path}/pc_raw_edges.csv", index=False)

raw_upstream_map, raw_downstream_map = build_adjacency_maps(raw_pc_edges, handle_undirected=False)
dbutils.fs.put(f"{pipeline_path}/raw_upstream_map.json", json.dumps(raw_upstream_map, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/raw_downstream_map.json", json.dumps(raw_downstream_map, indent=2), overwrite=True)

# Save filtered graph
rows = []
for edge in filtered_edges:
    if isinstance(edge, dict):
        rows.append({
            "from": edge["from"], "to": edge["to"],
            "edge_type": edge.get("edge_type", "directed"),
            "weight": edge.get("weight", 1.0),
            "abs_weight": abs(edge.get("weight", 1.0)),
            "source": edge.get("source", "pc_algorithm")
        })

cand_df = pd.DataFrame(rows)
if 'abs_weight' in cand_df.columns:
    cand_df = cand_df.sort_values("abs_weight", ascending=False)
cand_df.to_csv(f"{pipeline_path}/pc_causal_edges.csv", index=False)

print("="*80)
print("✓ PIPELINE A COMPLETE — All artifacts saved")
print("="*80)
print(f"\nSaved to: {pipeline_path}")
print(f"\nFinal Results:")
print(f"  - RAW edges: {len(raw_pc_edges)}")
print(f"  - FILTERED edges: {len(filtered_edges)}")
print(f"  - Removed: {len(review.get('removed', []))}")
print(f"  - Added: {len(review.get('added', []))}")
