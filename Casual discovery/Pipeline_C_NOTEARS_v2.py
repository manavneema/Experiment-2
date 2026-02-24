# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline C: NOTEARS-Based Causal Discovery
# MAGIC 
# MAGIC This notebook implements NOTEARS for causal discovery on cross-sectional data.
# MAGIC - Implements linear NOTEARS algorithm for DAG discovery
# MAGIC - Uses continuous optimization with acyclicity constraint
# MAGIC - Produces directed acyclic graph (DAG)
# MAGIC - Uses 65 days of training data (independent daily snapshots)

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
import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import functions as F

from scipy.optimize import minimize
from scipy.linalg import expm

# COMMAND ----------

# ===========================
# PIPELINE-SPECIFIC METHODS
# ===========================

class NOTEARSLinear:
    """
    Linear NOTEARS implementation for causal discovery.
    
    Based on: "DAGs with NO TEARS: Continuous Optimization for Structure Learning" (Zheng et al., 2018)
    Uses acyclicity constraint h(W) = tr(exp(W ⊙ W)) - d = 0.
    """
    
    def __init__(self, lambda1=0.0, lambda2=0.0, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=None, w_percentile=90):
        """
        Initialize NOTEARS.
        
        Args:
            lambda1: L1 regularization weight
            lambda2: L2 regularization weight
            max_iter: Maximum optimization iterations
            h_tol: Tolerance for acyclicity constraint
            rho_max: Maximum rho for augmented Lagrangian
            w_threshold: Fixed weight threshold (if None, use percentile)
            w_percentile: Percentile for adaptive thresholding
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.w_percentile = w_percentile
        self.W_est = None
        self.optimization_path = []
    
    def _loss(self, W, X):
        """Compute squared loss."""
        n, d = X.shape
        R = X - X @ W
        return 0.5 / n * np.trace(R.T @ R)
    
    def _h(self, W):
        """Acyclicity constraint: h(W) = tr(exp(W ⊙ W)) - d."""
        d = W.shape[0]
        M = W * W
        E = expm(M)
        return np.trace(E) - d
    
    def _h_grad(self, W):
        """Gradient of acyclicity constraint."""
        M = W * W
        E = expm(M)
        return 2 * W * E
    
    def _func(self, w_vec, X, rho, alpha, d):
        """Objective function."""
        W = w_vec.reshape(d, d)
        loss = self._loss(W, X)
        h_val = self._h(W)
        reg = self.lambda1 * np.sum(np.abs(W)) + 0.5 * self.lambda2 * np.sum(W ** 2)
        return loss + reg + alpha * h_val + 0.5 * rho * h_val ** 2
    
    def _grad(self, w_vec, X, rho, alpha, d):
        """Gradient of objective function."""
        n = X.shape[0]
        W = w_vec.reshape(d, d)
        R = X - X @ W
        loss_grad = -1.0 / n * X.T @ R
        h_val = self._h(W)
        h_grad = self._h_grad(W)
        reg_grad = self.lambda1 * np.sign(W) + self.lambda2 * W
        return (loss_grad + reg_grad + (alpha + rho * h_val) * h_grad).flatten()
    
    def fit(self, X):
        """
        Fit NOTEARS model to data.
        
        Args:
            X: Data matrix (n_samples, n_features)
        
        Returns:
            self
        """
        import time
        start_time = time.time()
        
        n, d = X.shape
        print(f"Fitting NOTEARS (n_samples={n}, n_features={d})")
        
        # Initialize W
        np.random.seed(42)
        W = np.random.randn(d, d) * 0.1
        np.fill_diagonal(W, 0)
        
        # Bounds (diagonal must be zero)
        bounds = [(0.0, 0.0) if i == j else (None, None) for i in range(d) for j in range(d)]
        w_vec = W.flatten()
        
        rho, alpha = 1.0, 0.0
        
        print(f"Starting optimization (max_iter={self.max_iter}, h_tol={self.h_tol})...")
        
        for iter_num in range(self.max_iter):
            iter_start = time.time()
            
            result = minimize(
                fun=lambda w: self._func(w, X, rho, alpha, d),
                x0=w_vec,
                jac=lambda w: self._grad(w, X, rho, alpha, d),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-8}
            )
            
            w_vec = result.x
            W = w_vec.reshape(d, d)
            
            h_val = self._h(W)
            loss_val = self._loss(W, X)
            iter_time = time.time() - iter_start
            
            self.optimization_path.append({
                'iteration': iter_num,
                'h_value': float(h_val),
                'alpha': float(alpha),
                'rho': float(rho),
                'loss': float(loss_val),
                'success': result.success,
                'iter_time_sec': iter_time
            })
            
            print(f"  Iter {iter_num}: h={h_val:.2e}, loss={loss_val:.4f}, rho={rho:.0e}, time={iter_time:.1f}s")
            
            if h_val <= self.h_tol:
                print(f"  ✓ Converged! h={h_val:.2e} <= {self.h_tol}")
                break
            elif rho >= self.rho_max:
                print(f"  ⚠ Max rho reached. Final h={h_val:.2e}")
                break
            else:
                alpha += rho * h_val
                rho *= 2
        
        print(f"Optimization completed in {time.time() - start_time:.1f}s ({iter_num + 1} iterations)")
        
        self.W_est = W.copy()
        self._threshold_weights()
        
        return self
    
    def _threshold_weights(self):
        """Apply weight threshold to create sparse adjacency matrix."""
        if self.W_est is None:
            return
        
        W_abs = np.abs(self.W_est)
        np.fill_diagonal(W_abs, 0)
        nonzero_weights = W_abs[W_abs > 0]
        
        print(f"\nWeight Distribution (before thresholding):")
        print(f"  Non-zero entries: {len(nonzero_weights)}")
        if len(nonzero_weights) > 0:
            print(f"  Min: {nonzero_weights.min():.6f}, Max: {nonzero_weights.max():.6f}")
            print(f"  Mean: {nonzero_weights.mean():.6f}, Median: {np.median(nonzero_weights):.6f}")
        
        # Determine threshold
        if self.w_threshold is not None:
            threshold = self.w_threshold
            print(f"\nUsing fixed threshold: {threshold:.6f}")
        elif len(nonzero_weights) > 0:
            threshold = np.percentile(nonzero_weights, self.w_percentile)
            print(f"\nUsing {self.w_percentile}th percentile threshold: {threshold:.6f}")
        else:
            threshold = 0
        
        # Apply threshold
        W_thresh = self.W_est.copy()
        W_thresh[np.abs(W_thresh) < threshold] = 0
        
        n_edges_after = np.sum(np.abs(W_thresh) > 0)
        print(f"Weight thresholding: {len(nonzero_weights)} -> {n_edges_after} edges")
        
        # Enforce DAG by keeping only stronger direction
        print(f"Enforcing DAG structure...")
        d = W_thresh.shape[0]
        bidirectional_removed = 0
        
        for i in range(d):
            for j in range(i + 1, d):
                w_ij, w_ji = abs(W_thresh[i, j]), abs(W_thresh[j, i])
                if w_ij > 0 and w_ji > 0:
                    if w_ij >= w_ji:
                        W_thresh[j, i] = 0
                    else:
                        W_thresh[i, j] = 0
                    bidirectional_removed += 1
        
        print(f"  Bidirectional pairs resolved: {bidirectional_removed}")
        print(f"  Final edges: {np.sum(np.abs(W_thresh) > 0)}")
        
        self.W_thresh = W_thresh
    
    def get_adjacency_matrix(self, thresholded=True):
        """Get adjacency matrix."""
        if thresholded and hasattr(self, 'W_thresh'):
            return self.W_thresh
        return self.W_est


def run_notears_algorithm(df, **notears_kwargs):
    """
    Run NOTEARS algorithm on data.
    
    Args:
        df: Preprocessed DataFrame
        **notears_kwargs: Parameters for NOTEARSLinear
    
    Returns:
        dict: Result containing edges, weight matrix, and metadata
    """
    print(f'Running NOTEARS (n_samples={len(df)}, n_features={len(df.columns)})')
    
    if df.isna().any().any():
        print("❌ Input contains NaN values")
        return {'method': 'notears-error', 'error': 'Input contains NaN values', 'W_est': None}
    
    if len(df) < 10:
        print(f"❌ Insufficient samples: {len(df)} < 10")
        return {'method': 'notears-error', 'error': f'Insufficient samples: {len(df)} < 10', 'W_est': None}
    
    n_samples, n_features = df.shape
    ratio = n_samples / n_features
    print(f"Sample-to-feature ratio: {ratio:.2f}")
    
    if ratio < 2.0:
        print(f"⚠️  WARNING: Low sample-to-feature ratio ({ratio:.2f}). NOTEARS may struggle.")
    
    try:
        X = df.values.astype(float)
        columns = df.columns.tolist()
        
        model = NOTEARSLinear(**notears_kwargs)
        model.fit(X)
        
        W_est = model.get_adjacency_matrix(thresholded=True)
        
        # Extract edges
        edges = []
        for i in range(len(columns)):
            for j in range(len(columns)):
                if i != j and abs(W_est[i, j]) > 0:
                    edges.append({
                        'from': columns[i],
                        'to': columns[j],
                        'weight': float(W_est[i, j]),
                        'abs_weight': float(abs(W_est[i, j])),
                        'type': 'directed'
                    })
        
        edges = sorted(edges, key=lambda x: x['abs_weight'], reverse=True)
        
        print(f"✓ NOTEARS successful: found {len(edges)} directed edges")
        
        return {
            'method': 'notears-success',
            'model': model,
            'W_est': W_est,
            'W_raw': model.W_est,
            'edges': edges,
            'columns': columns,
            'optimization_path': model.optimization_path,
            'final_h_value': model.optimization_path[-1]['h_value'] if model.optimization_path else None,
            'converged': model.optimization_path[-1]['h_value'] <= model.h_tol if model.optimization_path else False,
            'n_samples': n_samples,
            'n_features': n_features,
            'sample_feature_ratio': ratio
        }
        
    except Exception as e:
        import traceback
        print(f"❌ NOTEARS failed: {str(e)}")
        traceback.print_exc()
        return {
            'method': 'notears-error',
            'error': str(e),
            'W_est': None,
            'edges': None,
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

# COMMAND ----------

# ===========================
# CONFIGURATION
# ===========================

PIPELINE_NAME = "NOTEARS_Based"
METRICS_TABLE = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"
MAX_RUNS_TO_PIVOT = 65

# NOTEARS parameters
LAMBDA1 = 0.0
LAMBDA2 = 0.0
RHO_MAX = 1e16
MAX_ITER = 100
H_TOL = 1e-8
W_THRESHOLD = None
W_PERCENTILE = 90

# Feature selection parameters
TARGET_FEATURES = 35
CORRELATION_THRESHOLD = 0.98

# Artifact path
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"
pipeline_path = f"{path}/{PIPELINE_NAME}"

print(f"Pipeline C Configuration:")
print(f"  - Method: NOTEARS-Based")
print(f"  - Training Days: {MAX_RUNS_TO_PIVOT}")
print(f"  - Lambda1: {LAMBDA1}, Lambda2: {LAMBDA2}")
print(f"  - Max Iterations: {MAX_ITER}")
print(f"  - Weight Percentile: {W_PERCENTILE}")
print(f"  - Target Features: {TARGET_FEATURES}")
print(f"  - Artifact Path: {pipeline_path}")

# COMMAND ----------

# ===========================
# DRIVER CODE
# ===========================

print("="*80)
print("PIPELINE C: NOTEARS-BASED CAUSAL DISCOVERY")
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
    feature_sample_ratio=3.0
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
print(f"\nNOTEARS Requirements Check:")
print(f"  Samples: {n_samples}, Features: {n_features}")
print(f"  Sample-to-feature ratio: {n_samples / n_features:.2f}")

if final_features.isna().any().any():
    final_features = final_features.fillna(final_features.median())
    print("⚠️  Fixed NaN values with median imputation")

if n_features == 0:
    raise Exception("No features remaining after selection")

# Step 4: Generate blacklist
print("\n[Step 4] Generating blacklist...")
metric_cols = metrics_sdf.select("metric_name").distinct().rdd.flatMap(lambda x: x).collect()
HUMAN_PRIOR_BLACKLIST = generate_stage_blacklist(metric_cols)
print(f"  - Whitelist edges: {len(HUMAN_PRIOR_WHITELIST)}")
print(f"  - Blacklist edges: {len(HUMAN_PRIOR_BLACKLIST)}")

# Step 5: Run NOTEARS
print(f"\n[Step 5] Running NOTEARS...")
notears_result = run_notears_algorithm(
    final_features,
    lambda1=LAMBDA1,
    lambda2=LAMBDA2,
    max_iter=MAX_ITER,
    h_tol=H_TOL,
    rho_max=RHO_MAX,
    w_threshold=W_THRESHOLD,
    w_percentile=W_PERCENTILE
)

if notears_result['method'] == 'notears-error':
    raise Exception(f"Pipeline C Failed: {notears_result['error']}")

# Step 6: Extract raw edges
print(f"\n[Step 6] Extracting edges...")
raw_notears_edges = notears_result.get('edges', [])
print(f"NOTEARS discovered {len(raw_notears_edges)} edges (before filtering)")

if len(raw_notears_edges) == 0:
    raise Exception("NOTEARS found no edges")

# Step 7: Apply blacklist
print(f"\n[Step 7] Filtering blacklist...")
selected_feature_set = set(final_features.columns)
filtered_blacklist = [
    (a, b) for a, b in HUMAN_PRIOR_BLACKLIST 
    if a in selected_feature_set and b in selected_feature_set
]
blacklist_set = set(filtered_blacklist)
print(f"Applicable blacklist edges: {len(filtered_blacklist)}")

# Filter edges
filtered_edges = []
removed_by_blacklist = []
for edge in raw_notears_edges:
    if (edge['from'], edge['to']) in blacklist_set:
        removed_by_blacklist.append(edge)
        print(f"  Removed blacklisted edge: {edge['from']} -> {edge['to']}")
    else:
        filtered_edges.append(edge)

print(f"After blacklist: {len(raw_notears_edges)} -> {len(filtered_edges)} edges")

# Step 8: Apply whitelist
print(f"\n[Step 8] Applying whitelist...")
existing_edges = set((e['from'], e['to']) for e in filtered_edges)
added_from_whitelist = []

W_raw = notears_result.get('W_raw')
columns = notears_result.get('columns', [])
col_to_idx = {col: idx for idx, col in enumerate(columns)}
MIN_RAW_WEIGHT = 1e-6

for from_col, to_col in HUMAN_PRIOR_WHITELIST:
    if from_col in selected_feature_set and to_col in selected_feature_set:
        if (from_col, to_col) not in existing_edges:
            weight = None
            source = None
            
            # Try to recover from W_raw
            if W_raw is not None and from_col in col_to_idx and to_col in col_to_idx:
                i, j = col_to_idx[from_col], col_to_idx[to_col]
                raw_weight = W_raw[i, j]
                if abs(raw_weight) > MIN_RAW_WEIGHT:
                    weight = float(raw_weight)
                    source = 'whitelist_recovered'
                    print(f"    Recovered from W_raw: {from_col} -> {to_col} (weight={weight:.6f})")
            
            # Fallback to OLS
            if weight is None:
                weight = estimate_weight_via_regression(from_col, to_col, final_features)
                source = 'whitelist_estimated'
                print(f"    Estimated via OLS: {from_col} -> {to_col} (weight={weight:.6f})")
            
            filtered_edges.append({
                'from': from_col,
                'to': to_col,
                'weight': weight,
                'abs_weight': abs(weight),
                'type': 'directed',
                'source': source
            })
            added_from_whitelist.append((from_col, to_col, source, weight))

n_recovered = sum(1 for x in added_from_whitelist if x[2] == 'whitelist_recovered')
n_estimated = sum(1 for x in added_from_whitelist if x[2] == 'whitelist_estimated')
print(f"\nWhitelist Summary:")
print(f"  - Edges recovered from W_raw: {n_recovered}")
print(f"  - Edges estimated via OLS: {n_estimated}")
print(f"  - Total added: {len(added_from_whitelist)}")
print(f"Final edge count: {len(filtered_edges)}")

# Step 9: Visualize
print(f"\n[Step 9] Visualizing graphs...")
G_raw = visualize_dag(raw_notears_edges, title="Pipeline C: NOTEARS RAW DAG")
G = visualize_dag(filtered_edges, title="Pipeline C: NOTEARS DAG (After Blacklist/Whitelist)")

print("="*80)
print("PIPELINE C: SUCCESSFUL COMPLETION")
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
    "method": "notears-based",
    "data_type": "cross-sectional",
    "status": "SUCCESS",
    "training_days": MAX_RUNS_TO_PIVOT,
    "preprocess_meta": preprocess_meta,
    "feature_selection_log": feature_selection_log,
    "notears_result": {
        "method": notears_result["method"],
        "lambda1": LAMBDA1,
        "lambda2": LAMBDA2,
        "w_threshold": W_THRESHOLD,
        "max_iter": MAX_ITER,
        "n_samples": notears_result["n_samples"],
        "n_features": notears_result["n_features"],
        "sample_feature_ratio": notears_result["sample_feature_ratio"],
        "converged": notears_result.get("converged", False),
        "final_h_value": notears_result.get("final_h_value"),
        "optimization_iterations": len(notears_result.get("optimization_path", [])),
        "raw_edges_found": len(raw_notears_edges)
    },
    "edge_stats": {
        "raw_edges": len(raw_notears_edges),
        "removed_by_blacklist": len(removed_by_blacklist),
        "added_from_whitelist": len(added_from_whitelist),
        "final_edges": len(filtered_edges),
        "edge_type": "directed",
        "orientation_method": "notears_optimization"
    },
    "raw_notears_edges": raw_notears_edges,
    "filtered_edges": filtered_edges,
    "blacklist_filtering": {
        "blacklist_edges_applicable": len(filtered_blacklist),
        "edges_removed": [(e['from'], e['to']) for e in removed_by_blacklist],
        "whitelist_edges_added": len(added_from_whitelist),
        "whitelist_recovered_from_W_raw": n_recovered,
        "whitelist_estimated_via_ols": n_estimated,
        "whitelist_details": [(x[0], x[1], x[2], float(x[3])) for x in added_from_whitelist]
    },
    "final_graph_stats": {
        "total_edges": len(filtered_edges),
        "nodes_with_parents": len(upstream_map),
        "nodes_with_children": len(downstream_map),
        "is_dag": True
    }
}

# Save core artifacts
final_features.to_csv(f"{pipeline_path}/causal_metrics_matrix.csv")
dbutils.fs.put(f"{pipeline_path}/causal_artifacts.json", json.dumps(artifacts, indent=2, default=str), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/baseline_stats.json", json.dumps(baseline_stats, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/upstream_map.json", json.dumps(upstream_map, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/downstream_map.json", json.dumps(downstream_map, indent=2), overwrite=True)

# Save NOTEARS-specific artifacts
if notears_result.get('W_est') is not None:
    W_df = pd.DataFrame(notears_result['W_est'], index=notears_result['columns'], columns=notears_result['columns'])
    W_df.to_csv(f"{pipeline_path}/notears_weight_matrix.csv")

if notears_result.get('optimization_path'):
    pd.DataFrame(notears_result['optimization_path']).to_csv(f"{pipeline_path}/notears_optimization_path.csv", index=False)

# Save raw graph
raw_rows = []
for edge in raw_notears_edges:
    if isinstance(edge, dict):
        raw_rows.append({
            "from": edge["from"], "to": edge["to"],
            "weight": edge.get("weight", 0.0),
            "abs_weight": edge.get("abs_weight", 0.0),
            "edge_type": edge.get("type", "directed"),
            "source": "notears_raw"
        })

pd.DataFrame(raw_rows).sort_values("abs_weight", ascending=False).to_csv(f"{pipeline_path}/notears_raw_edges.csv", index=False)

raw_upstream_map, raw_downstream_map = build_adjacency_maps(raw_notears_edges, handle_undirected=False)
dbutils.fs.put(f"{pipeline_path}/raw_upstream_map.json", json.dumps(raw_upstream_map, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/raw_downstream_map.json", json.dumps(raw_downstream_map, indent=2), overwrite=True)

# Save filtered graph
rows = []
for edge in filtered_edges:
    if isinstance(edge, dict):
        rows.append({
            "from": edge["from"], "to": edge["to"],
            "weight": edge.get("weight", 0.0),
            "abs_weight": edge.get("abs_weight", 0.0),
            "edge_type": edge.get("type", "directed"),
            "source": edge.get("source", "notears")
        })

pd.DataFrame(rows).sort_values("abs_weight", ascending=False).to_csv(f"{pipeline_path}/notears_causal_edges.csv", index=False)

print("="*80)
print("✓ PIPELINE C COMPLETE — All artifacts saved")
print("="*80)
print(f"\nSaved to: {pipeline_path}")
print(f"\nFinal Results:")
print(f"  - NOTEARS converged: {notears_result.get('converged', False)}")
print(f"  - RAW edges: {len(raw_notears_edges)}")
print(f"  - FILTERED edges: {len(filtered_edges)}")
print(f"  - Removed by blacklist: {len(removed_by_blacklist)}")
print(f"  - Added from whitelist: {len(added_from_whitelist)}")
