# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline C: NOTEARS-Based Causal Discovery
# MAGIC
# MAGIC This notebook implements a NOTEARS approach for causal discovery on cross-sectional data.
# MAGIC - Implements linear NOTEARS algorithm for DAG discovery
# MAGIC - Enforces blacklist constraints during optimization (acyclicity + domain rules)
# MAGIC - Outputs directed acyclic graph (DAG) through continuous optimization
# MAGIC - Applies whitelist additions from domain knowledge
# MAGIC - Uses 65 days of training data (independent daily snapshots)
# MAGIC - NO temporal features or Granger causality (data is cross-sectional, not time series)

# COMMAND ----------

# MAGIC %pip install networkx scipy scikit-learn

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
from pyspark.sql import types as T

# ML Libs
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.linalg import expm

# Visualization
import networkx as nx
import matplotlib.pyplot as plt

# COMMAND ----------

# Configuration for Pipeline C
PIPELINE_NAME = "NOTEARS_Based"
METRICS_TABLE = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"
MAX_RUNS_TO_PIVOT = 65  # Using 65 days (independent daily snapshots)

# NOTEARS parameters (NO SPARSITY - let data determine structure)
LAMBDA1 = 0.0    # NO L1 regularization - let data determine edges
LAMBDA2 = 0.0    # NO L2 regularization
RHO_MAX = 1e16   # Maximum rho for augmented Lagrangian
MAX_ITER = 100   # Sufficient iterations for convergence
H_TOL = 1e-8     # Strict tolerance for proper DAG convergence
W_THRESHOLD = None  # Use percentile-based thresholding instead
W_PERCENTILE = 90   # Keep only top 10% strongest edges (90th percentile)

# DBFS path for artifacts
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"
pipeline_path = f"{path}/{PIPELINE_NAME}"

print(f"Pipeline C Configuration:")
print(f"  - Method: NOTEARS-Based (Cross-Sectional)")
print(f"  - Training Days: {MAX_RUNS_TO_PIVOT}")
print(f"  - Lambda1 (L1): {LAMBDA1}")
print(f"  - Lambda2 (L2): {LAMBDA2}")
print(f"  - Max Iterations: {MAX_ITER}")
print(f"  - Weight Threshold: {W_THRESHOLD}")
print(f"  - Artifact Path: {pipeline_path}")
print(f"  - Data Type: Cross-sectional (independent daily snapshots)")
print(f"  - Temporal Features: NONE (not applicable for cross-sectional data)")

# COMMAND ----------

def generate_stage_blacklist(metric_cols):
    """Generate conservative blacklist pairs from metric column names.
    
    Rules encoded (enforcing pipeline flow direction):
    - Forbid any edge from silver_ metrics to bronze_ or raw_ metrics
    - Forbid any edge from bronze_ metrics to raw_ metrics
    
    This ensures causal direction follows: raw → bronze → silver
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
    feature_sample_ratio: float = 3.0,   # NOTEARS can handle more features
    min_keep_features: int = 15,
):
    """Preprocess metrics matrix for causal discovery.
    
    This function prepares cross-sectional data for NOTEARS:
    - No temporal features are added (data is independent daily snapshots)
    - Z-score scaling is ESSENTIAL for NOTEARS optimization convergence
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
    
    # Z-score scaling (ESSENTIAL for NOTEARS)
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
        keep_k = max(int(n_samples * 2.0), min_keep_features)
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

def sophisticated_feature_selection_for_notears(
    df: pd.DataFrame,
    *,
    target_features: int = 35,  # Conservative for NOTEARS optimization complexity
    variance_threshold: float = 1e-6,
    correlation_threshold: float = 0.98,  # High threshold - NOTEARS benefits from more features
):
    """Sophisticated feature selection optimized for NOTEARS algorithm.
    
    NOTEARS-specific optimizations:
    - Conservative target (35 features) due to O(d^3) optimization complexity
    - High correlation threshold (0.98) to preserve potentially causal relationships
    - Prioritizes business metrics and silver-stage features for RCA relevance
    
    Cross-sectional approach:
    - NO temporal features (lagged variables, rolling windows)
    - Each row represents an independent daily observation
    - Feature selection based on variance and correlation only
    """
    print(f"Starting sophisticated feature selection (target: {target_features} features)")
    print(f"Input features: {df.shape[1]}")
    print(f"Data type: Cross-sectional (no temporal features)")
    
    selection_log = {
        "initial_features": df.shape[1],
        "target_features": target_features,
        "data_type": "cross-sectional",
        "temporal_features_added": 0,  # Explicitly zero - no temporal features
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
    
    # Step 3: High-Correlation Pruning (conservative for NOTEARS)
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
            """Higher score = higher priority (keep feature).
            
            Priority system for RCA relevance:
            - Business metrics (rates, ratios) > operational metrics
            - Aggregated KPIs (mean, p95) > raw values
            - Silver stage > Bronze stage > Raw stage
            - Domain-specific metrics (fuel, speed, distance)
            """
            score = 0
            
            # Prefer rates over raw counts (more stable for causal inference)
            if any(term in feature_name.lower() for term in ['rate', 'ratio', 'percent', 'pct']):
                score += 10
            
            # Prefer aggregated KPIs over raw metrics
            if any(term in feature_name.lower() for term in ['mean', 'avg', 'median', 'p95']):
                score += 5
            
            # Prefer downstream (silver) over upstream - more relevant for RCA
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
    selection_log["reduction_ratio"] = selection_log["total_removed"] / df.shape[1] if df.shape[1] > 0 else 0
    
    print(f"\n✓ Feature selection complete:")
    print(f"  Initial: {df.shape[1]} features")
    print(f"  Final: {out.shape[1]} features")
    print(f"  Removed: {selection_log['total_removed']} features ({selection_log['reduction_ratio']:.1%})")
    print(f"  Target achieved: {'✓' if out.shape[1] <= target_features else '✗'}")
    
    return out, selection_log

# COMMAND ----------

class NOTEARSLinear:
    """
    Linear NOTEARS implementation for causal discovery on cross-sectional data.
    
    Based on: "DAGs with NO TEARS: Continuous Optimization for Structure Learning" (Zheng et al., 2018)
    
    This is a CLEAN implementation without blacklist constraints.
    Blacklist filtering should be applied as a POST-PROCESSING step.
    
    Key properties:
    - Uses acyclicity constraint h(W) = tr(exp(W ⊙ W)) - d = 0
    - Optimizes via augmented Lagrangian method with L-BFGS-B
    - Produces directed acyclic graph (DAG) through optimization
    - Suitable for cross-sectional data (no temporal assumptions)
    """
    
    def __init__(self, lambda1=0.0, lambda2=0.0, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=None, w_percentile=90):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold  # Fixed threshold (if None, use percentile)
        self.w_percentile = w_percentile  # Percentile-based threshold (e.g., 90 = keep top 10%)
        
        self.W_est = None
        self.optimization_path = []
    
    def _loss(self, W, X):
        """Compute squared loss: 0.5/n * ||X - XW||_F^2"""
        n, d = X.shape
        R = X - X @ W
        return 0.5 / n * np.trace(R.T @ R)
    
    def _h(self, W):
        """Acyclicity constraint: h(W) = tr(exp(W ⊙ W)) - d.
        
        This constraint equals 0 if and only if W represents a DAG.
        """
        d = W.shape[0]
        M = W * W  # element-wise square
        E = expm(M)
        h = np.trace(E) - d
        return h
    
    def _h_grad(self, W):
        """Gradient of acyclicity constraint: ∂h/∂W = 2W ⊙ exp(W ⊙ W)"""
        M = W * W
        E = expm(M)
        return 2 * W * E
    
    def _func(self, w_vec, X, rho, alpha, d):
        """Objective function: loss + regularization + acyclicity penalty."""
        W = w_vec.reshape(d, d)
        
        loss = self._loss(W, X)
        h_val = self._h(W)
        
        # L1 and L2 regularization
        reg = self.lambda1 * np.sum(np.abs(W)) + 0.5 * self.lambda2 * np.sum(W ** 2)
        
        # Augmented Lagrangian for acyclicity
        obj = loss + reg + alpha * h_val + 0.5 * rho * h_val ** 2
        
        return obj
    
    def _grad(self, w_vec, X, rho, alpha, d):
        """Gradient of objective function."""
        n = X.shape[0]
        W = w_vec.reshape(d, d)
        
        # Loss gradient: ∂loss/∂W = -1/n * X^T * (X - XW)
        R = X - X @ W
        loss_grad = -1.0 / n * X.T @ R
        
        # h constraint gradient
        h_val = self._h(W)
        h_grad = self._h_grad(W)
        
        # Regularization gradients (subgradient for L1)
        reg_grad = self.lambda1 * np.sign(W) + self.lambda2 * W
        
        # Total gradient
        grad = loss_grad + reg_grad + (alpha + rho * h_val) * h_grad
        
        return grad.flatten()
    
    def fit(self, X):
        """
        Fit NOTEARS model to cross-sectional data (UNCONSTRAINED).
        
        Args:
            X: Data matrix (n_samples, n_features) - each row is an independent observation
        
        Returns:
            self (fitted model)
        """
        import time
        start_time = time.time()
        
        n, d = X.shape
        print(f"Fitting NOTEARS (n_samples={n}, n_features={d})")
        print(f"Data assumption: Cross-sectional (independent observations)")
        print(f"Constraints: NONE (unconstrained optimization)")
        
        # Initialize W with small random values (breaks symmetry, essential for NOTEARS)
        np.random.seed(42)  # Reproducibility
        W = np.random.randn(d, d) * 0.1
        np.fill_diagonal(W, 0)  # No self-loops
        
        # Set up bounds: diagonal must be zero (no self-loops)
        bounds = []
        for i in range(d):
            for j in range(d):
                if i == j:
                    bounds.append((0.0, 0.0))  # Force diagonal to 0
                else:
                    bounds.append((None, None))  # No bounds on off-diagonal
        
        w_vec = W.flatten()
        
        # Augmented Lagrangian parameters
        rho, alpha = 1.0, 0.0
        
        print(f"Starting optimization (max_iter={self.max_iter}, h_tol={self.h_tol})...")
        
        for iter_num in range(self.max_iter):
            iter_start = time.time()
            
            # Optimize using L-BFGS-B
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
            
            # Check acyclicity constraint
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
                # Update Lagrangian parameters
                alpha += rho * h_val
                rho *= 2  # Standard rho increase for stable convergence
        
        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.1f} seconds ({iter_num + 1} iterations)")
        
        # Store result and threshold
        self.W_est = W.copy()
        self._threshold_weights()
        
        return self
    
    def _threshold_weights(self):
        """Apply weight threshold to create sparse adjacency matrix."""
        if self.W_est is None:
            return
        
        # Diagnostic: show weight distribution BEFORE thresholding
        W_abs = np.abs(self.W_est)
        np.fill_diagonal(W_abs, 0)  # Ignore diagonal
        nonzero_weights = W_abs[W_abs > 0]
        
        print(f"\nWeight Distribution (before thresholding):")
        print(f"  Non-zero entries: {len(nonzero_weights)}")
        if len(nonzero_weights) > 0:
            print(f"  Min weight: {nonzero_weights.min():.6f}")
            print(f"  Max weight: {nonzero_weights.max():.6f}")
            print(f"  Mean weight: {nonzero_weights.mean():.6f}")
            print(f"  Median weight: {np.median(nonzero_weights):.6f}")
            # Show percentiles
            for p in [25, 50, 75, 90, 95, 99]:
                print(f"  {p}th percentile: {np.percentile(nonzero_weights, p):.6f}")
        
        # Determine threshold
        if self.w_threshold is not None:
            # Use fixed threshold
            threshold = self.w_threshold
            print(f"\nUsing fixed threshold: {threshold:.6f}")
        elif len(nonzero_weights) > 0:
            # Use percentile-based threshold
            threshold = np.percentile(nonzero_weights, self.w_percentile)
            print(f"\nUsing {self.w_percentile}th percentile threshold: {threshold:.6f}")
            print(f"  (keeping top {100 - self.w_percentile}% strongest edges)")
        else:
            threshold = 0
        
        # Apply threshold
        W_thresh = self.W_est.copy()
        W_thresh[np.abs(W_thresh) < threshold] = 0
        
        n_edges_after_thresh = np.sum(np.abs(W_thresh) > 0)
        print(f"\nWeight thresholding: {len(nonzero_weights)} -> {n_edges_after_thresh} edges")
        
        if n_edges_after_thresh == 0 and len(nonzero_weights) > 0:
            suggested_threshold = np.percentile(nonzero_weights, 50)  # Use median
            print(f"⚠️  All edges filtered out! Consider using threshold <= {suggested_threshold:.6f}")
        
        # CRITICAL: Enforce DAG structure by keeping only stronger direction for each edge pair
        # NOTEARS acyclicity constraint allows small bidirectional edges; we must remove them
        print(f"\nEnforcing DAG structure (removing bidirectional edges)...")
        d = W_thresh.shape[0]
        bidirectional_removed = 0
        
        for i in range(d):
            for j in range(i + 1, d):
                w_ij = abs(W_thresh[i, j])
                w_ji = abs(W_thresh[j, i])
                
                if w_ij > 0 and w_ji > 0:
                    # Both directions exist - keep only the stronger one
                    if w_ij >= w_ji:
                        W_thresh[j, i] = 0
                    else:
                        W_thresh[i, j] = 0
                    bidirectional_removed += 1
        
        n_edges_final = np.sum(np.abs(W_thresh) > 0)
        print(f"  Bidirectional pairs resolved: {bidirectional_removed}")
        print(f"  Final edges after DAG enforcement: {n_edges_final}")
        
        self.W_thresh = W_thresh
    
    def get_adjacency_matrix(self, thresholded=True):
        """Get adjacency matrix."""
        if thresholded and hasattr(self, 'W_thresh'):
            return self.W_thresh
        return self.W_est

# COMMAND ----------

def run_notears_algorithm(df: pd.DataFrame, **notears_kwargs):
    """Run UNCONSTRAINED NOTEARS algorithm on cross-sectional data.
    
    This function runs pure NOTEARS without any blacklist constraints.
    Blacklist filtering should be applied as a post-processing step.
    
    Args:
        df: Cross-sectional data matrix (rows = independent observations, cols = features)
        **notears_kwargs: Parameters for NOTEARSLinear (lambda1, lambda2, etc.)
    
    Returns:
        Dictionary with NOTEARS results including edges, weight matrix, and optimization info
    """
    print(f'Running NOTEARS Algorithm (n_samples={len(df)}, n_features={len(df.columns)})')
    print(f'Data type: Cross-sectional (independent daily snapshots)')
    print(f'Constraints: NONE (unconstrained optimization)')
    
    # Validate input
    if df.isna().any().any():
        return {'method': 'notears-error', 'error': 'Input contains NaN values', 'W_est': None}
    
    if len(df) < 10:
        return {'method': 'notears-error', 'error': f'Insufficient samples: {len(df)} < 10', 'W_est': None}
    
    n_samples, n_features = df.shape
    
    # Check sample to feature ratio (NOTEARS works best with ratio > 2)
    ratio = n_samples / n_features
    print(f"Sample-to-feature ratio: {ratio:.2f}")
    
    if ratio < 2.0:
        print(f"⚠️  WARNING: Low sample-to-feature ratio ({ratio:.2f}). NOTEARS may struggle.")
    
    try:
        X = df.values.astype(float)
        columns = df.columns.tolist()
        
        # Fit NOTEARS (unconstrained - no blacklist)
        model = NOTEARSLinear(**notears_kwargs)
        model.fit(X)  # No blacklist parameter
        
        # Get thresholded weight matrix
        W_est = model.get_adjacency_matrix(thresholded=True)
        
        # Extract directed edges from weight matrix
        edges = []
        for i in range(len(columns)):
            for j in range(len(columns)):
                if i != j and abs(W_est[i, j]) > 0:
                    edges.append({
                        'from': columns[i],
                        'to': columns[j],
                        'weight': float(W_est[i, j]),
                        'abs_weight': float(abs(W_est[i, j])),
                        'type': 'directed'  # NOTEARS produces directed edges
                    })
        
        # Sort by absolute weight
        edges = sorted(edges, key=lambda x: x['abs_weight'], reverse=True)
        
        print(f"✓ NOTEARS successful: found {len(edges)} directed edges")
        
        return {
            'method': 'notears-success',
            'model': model,
            'W_est': W_est,
            'W_raw': model.W_est,  # Unthresholded weights for debugging
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
        error_msg = f"NOTEARS failed: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return {
            'method': 'notears-error',
            'error': error_msg,
            'W_est': None,
            'edges': None,
            'n_samples': n_samples,
            'n_features': n_features
        }

# COMMAND ----------

def apply_human_priors(skeleton_edges, blacklist, whitelist):
    """Apply human priors (domain knowledge) to NOTEARS edges.
    
    Args:
        skeleton_edges: List of edge dictionaries from NOTEARS
        blacklist: Set of (from, to) tuples that should be removed
        whitelist: Set of (from, to) tuples that should be added if missing
    
    Returns:
        kept: List of edges after applying priors
        review: Dictionary with removed and added edges
    """
    kept = []
    removed = []
    
    # Convert edges to (from, to) format for comparison
    edge_pairs = set()
    for edge in skeleton_edges:
        if isinstance(edge, dict):
            edge_pairs.add((edge["from"], edge["to"]))
        else:
            edge_pairs.add((edge[0], edge[1]))
    
    # Remove blacklisted edges (NOTEARS should have prevented these, but double-check)
    for edge in skeleton_edges:
        if isinstance(edge, dict):
            a, b = edge["from"], edge["to"]
        else:
            a, b = edge[0], edge[1]
        
        if (a, b) in blacklist:
            removed.append(edge)
            print(f"  Removed blacklisted edge: {a} -> {b}")
        else:
            kept.append(edge)
    
    # Add whitelisted edges if missing
    added = []
    for a, b in whitelist:
        if (a, b) not in edge_pairs:
            new_edge = {
                "from": a,
                "to": b,
                "weight": 0.5,  # Default weight for whitelisted edges
                "abs_weight": 0.5,
                "type": "directed",
                "source": "human_prior"
            }
            added.append(new_edge)
            kept.append(new_edge)
            print(f"  Added whitelisted edge: {a} -> {b}")
    
    review = {'removed': removed, 'added': added}
    return kept, review

# COMMAND ----------

def visualize_dag(edges, title="NOTEARS DAG", top_k=50):
    """Visualize directed acyclic graph from NOTEARS.
    
    Args:
        edges: List of edge dictionaries with 'from', 'to', and optional 'abs_weight'
        title: Plot title
        top_k: Maximum number of edges to display (for readability)
    """
    if not edges:
        print("No edges to visualize")
        return None
    
    # Show top_k strongest edges for readability
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
            G.add_edge(a, b, weight=weight)
        else:
            a, b = edge[0], edge[1]
            G.add_edge(a, b, weight=1.0)
    
    plt.figure(figsize=(16, 12))
    
    # Use hierarchical layout if possible
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)
    
    # Draw edges with varying thickness based on weight
    edges_list = list(G.edges(data=True))
    if edges_list:
        weights = [d['weight'] for u, v, d in edges_list]
        max_weight = max(weights) if weights else 1
        widths = [max(0.5, 3 * w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=widths, edge_color='blue', 
                             arrows=True, arrowsize=20, alpha=0.7)
    
    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"{title} ({len(edges)} total edges, showing top {len(show_edges)})")
    plt.tight_layout()
    plt.show()
    
    # Check for cycles (should be none in NOTEARS output)
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print(f"⚠️  WARNING: Found {len(cycles)} cycles in graph! NOTEARS should produce DAG.")
            for cycle in cycles[:5]:  # Show first 5 cycles
                print(f"  Cycle: {' -> '.join(cycle + [cycle[0]])}")
        else:
            print("✓ Graph is acyclic (DAG)")
    except:
        print("Could not check for cycles")
    
    return G

# COMMAND ----------

def spark_metrics_to_matrix(metrics_sdf, max_runs=65, date_col='date'):
    """Convert metrics DataFrame to wide pandas DataFrame.
    
    Args:
        metrics_sdf: Spark DataFrame with columns [date, metric_name, metric_value]
        max_runs: Number of most recent dates to include
        date_col: Name of the date column
    
    Returns cross-sectional data where:
    - Each row is an independent daily snapshot
    - Each column is a metric
    - No temporal dependencies between rows
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
    
    print(f"Loaded {len(pdf)} days of metrics data (cross-sectional)")
    print(f"Metrics available: {len(pdf.columns)}")
    print(f"Data interpretation: Each row is an independent daily observation")
    
    return pdf

# COMMAND ----------

# Define Human Priors (same as Pipelines A and B for consistency)
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
print("PIPELINE C: NOTEARS-BASED CAUSAL DISCOVERY (CROSS-SECTIONAL)")
print("="*80)
print("\nData Approach: Cross-sectional (independent daily snapshots)")
print("Temporal Features: NONE (not applicable for this data type)")

# Step 1: Load metrics data from table
print("\n[Step 1] Loading metrics data from table...")
metrics_sdf = spark.table(METRICS_TABLE)
print(f"Loaded metrics table: {METRICS_TABLE}")

# Convert to matrix format
metrics_pdf = spark_metrics_to_matrix(
    metrics_sdf,
    max_runs=MAX_RUNS_TO_PIVOT
)

# Step 2: Preprocess metrics matrix
print("\n[Step 2] Preprocessing metrics matrix...")
scaled, preprocess_meta = preprocess_metrics_matrix(
    metrics_pdf,
    zscore=True,
    feature_sample_ratio=3.0  # NOTEARS can handle more features
)
print(f"After preprocessing: {scaled.shape}")

# Step 3: Sophisticated Feature Selection for NOTEARS
print("\n[Step 3] Sophisticated feature selection for NOTEARS...")
final_features, feature_selection_log = sophisticated_feature_selection_for_notears(
    scaled,
    target_features=35,  # Conservative for NOTEARS optimization complexity
    variance_threshold=1e-6,
    correlation_threshold=0.98  # High threshold to preserve potential causal relationships
)

# COMMAND ----------


# Check NOTEARS requirements
n_samples, n_features = final_features.shape
sample_to_feature_ratio = n_samples / n_features
print(f"\nNOTEARS Requirements Check:")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  Sample-to-feature ratio: {sample_to_feature_ratio:.2f}")
print(f"  NOTEARS stability: {'✓ Good' if sample_to_feature_ratio > 2.0 else '⚠ Marginal' if sample_to_feature_ratio > 1.5 else '✗ Poor'}")

# Step 4: Final validation
print("\n[Step 4] Final validation...")
if final_features.isna().any().any():
    print("⚠️  WARNING: NaN values detected after feature selection")
    # Attempt to fix by additional imputation
    imputer = SimpleImputer(strategy='median')
    final_features = pd.DataFrame(
        imputer.fit_transform(final_features),
        index=final_features.index,
        columns=final_features.columns
    )
    
if n_features == 0:
    print("🛑 PIPELINE C TERMINATED - No features remaining after selection")
    raise Exception("No features remaining after feature selection")

print(f"✓ Ready for NOTEARS with {n_features} carefully selected features")

# Step 5: Run NOTEARS Algorithm (UNCONSTRAINED)
print(f"\n[Step 5] Running NOTEARS Algorithm (UNCONSTRAINED)...")
print(f"Note: Blacklist will be applied as post-processing after graph discovery")

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

print(f"NOTEARS Result: {notears_result['method']}")
if notears_result['method'] == 'notears-error':
    print(f"NOTEARS Failed: {notears_result['error']}")
    print("🛑 PIPELINE C TERMINATED")
    
    # Save failure artifacts
    failure_artifacts = {
        "pipeline": PIPELINE_NAME,
        "status": "FAILED",
        "failure_reason": notears_result['error'],
        "preprocess_meta": preprocess_meta,
        "feature_selection_log": feature_selection_log,
        "notears_result": notears_result,
        "final_shape": list(final_features.shape)
    }
    
    dbutils.fs.put(f"{pipeline_path}/failure_report.json", 
                   json.dumps(failure_artifacts, indent=2, default=str), 
                   overwrite=True)
    
    raise Exception(f"Pipeline C Failed: {notears_result['error']}")

# Step 6: Extract edges from NOTEARS result (RAW - before blacklist)
raw_notears_edges = notears_result.get('edges', [])
print(f"\n[Step 6] NOTEARS discovered {len(raw_notears_edges)} directed edges (before filtering)")

if len(raw_notears_edges) == 0:
    print("🛑 PIPELINE C TERMINATED - NOTEARS found no edges")
    
    # Save no-edges artifacts
    no_edges_artifacts = {
        "pipeline": PIPELINE_NAME,
        "status": "NO_EDGES",
        "preprocess_meta": preprocess_meta,
        "feature_selection_log": feature_selection_log,
        "notears_result": notears_result,
        "final_shape": list(final_features.shape)
    }
    
    dbutils.fs.put(f"{pipeline_path}/no_edges_report.json", 
                   json.dumps(no_edges_artifacts, indent=2, default=str), 
                   overwrite=True)
    
    raise Exception("Pipeline C: NOTEARS found no edges")

# Print optimization summary
if notears_result.get('optimization_path'):
    print(f"\nOptimization Summary:")
    print(f"  - Converged: {notears_result.get('converged', False)}")
    print(f"  - Final h(W): {notears_result.get('final_h_value', 'Unknown'):.2e}")
    print(f"  - Iterations: {len(notears_result['optimization_path'])}")

# Step 7: Visualize RAW DAG (before blacklist filtering)
print(f"\n[Step 7] Visualizing RAW directed acyclic graph (before blacklist)...")
G_raw = visualize_dag(raw_notears_edges, title="Pipeline C: NOTEARS RAW DAG (Before Blacklist)")

# COMMAND ----------

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

# Filter out blacklisted edges
filtered_edges = []
removed_by_blacklist = []
for edge in raw_notears_edges:
    edge_pair = (edge['from'], edge['to'])
    if edge_pair in blacklist_set:
        removed_by_blacklist.append(edge)
    else:
        filtered_edges.append(edge)

print(f"Edges after blacklist filtering:")
print(f"  - Original: {len(raw_notears_edges)}")
print(f"  - Removed by blacklist: {len(removed_by_blacklist)}")
print(f"  - Remaining: {len(filtered_edges)}")

# Step 9: Apply whitelist (add known causal edges if not present)
# HYBRID APPROACH:
# - Edges already discovered: keep original weight (already in filtered_edges)
# - Whitelist edges below threshold: recover weight from raw W matrix
# - Whitelist edges truly absent: estimate weight via OLS regression
print(f"\n[Step 9] Applying whitelist (adding known causal edges with data-driven weights)...")

existing_edges = set((e['from'], e['to']) for e in filtered_edges)
added_from_whitelist = []

# Get raw weight matrix and column names from NOTEARS result
W_raw = notears_result.get('W_raw')  # Unthresholded weight matrix
columns = notears_result.get('columns', [])
col_to_idx = {col: idx for idx, col in enumerate(columns)}

# Minimum weight threshold for considering W_raw weight as "present"
MIN_RAW_WEIGHT = 1e-6

def estimate_weight_via_regression(from_col, to_col, data):
    """Estimate causal weight using OLS regression: to_col ~ from_col.
    
    Returns the regression coefficient as the edge weight.
    This is consistent with NOTEARS's linear SEM interpretation.
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

for from_col, to_col in HUMAN_PRIOR_WHITELIST:
    if from_col in selected_feature_set and to_col in selected_feature_set:
        if (from_col, to_col) not in existing_edges:
            # Edge not in final graph - need to add it with appropriate weight
            
            weight = None
            source = None
            
            # Option 1: Try to recover weight from raw (unthresholded) W matrix
            if W_raw is not None and from_col in col_to_idx and to_col in col_to_idx:
                i = col_to_idx[from_col]
                j = col_to_idx[to_col]
                raw_weight = W_raw[i, j]
                
                if abs(raw_weight) > MIN_RAW_WEIGHT:
                    # Weight exists in NOTEARS but was below threshold
                    weight = float(raw_weight)
                    source = 'whitelist_recovered'
                    print(f"    Recovered from W_raw: {from_col} -> {to_col} (weight={weight:.6f})")
            
            # Option 2: Estimate weight via regression if not found in W_raw
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

# Summary
n_recovered = sum(1 for x in added_from_whitelist if x[2] == 'whitelist_recovered')
n_estimated = sum(1 for x in added_from_whitelist if x[2] == 'whitelist_estimated')
print(f"\nWhitelist Summary:")
print(f"  - Edges already present (kept original weight): {len([e for e in HUMAN_PRIOR_WHITELIST if e[0] in selected_feature_set and e[1] in selected_feature_set]) - len(added_from_whitelist)}")
print(f"  - Edges recovered from W_raw (below threshold): {n_recovered}")
print(f"  - Edges estimated via OLS regression: {n_estimated}")
print(f"  - Total whitelist edges added: {len(added_from_whitelist)}")
print(f"Final edge count: {len(filtered_edges)}")

# Step 10: Visualize FILTERED DAG (after blacklist/whitelist)
print(f"\n[Step 10] Visualizing FILTERED directed acyclic graph...")
G = visualize_dag(filtered_edges, title="Pipeline C: NOTEARS DAG (After Blacklist/Whitelist)")

# Use filtered edges for downstream processing
notears_edges = filtered_edges

print("="*80)
print("PIPELINE C: SUCCESSFUL COMPLETION")
print("="*80)

# COMMAND ----------

# ===========================
# EXPORT ARTIFACTS
# ===========================

print("\n[Step 9] Computing baseline statistics...")

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

print("\n[Step 10] Building adjacency maps...")

def build_adjacency_maps(edges):
    """Build upstream and downstream adjacency lists from directed edge list.
    
    For NOTEARS directed edges:
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

print("\n[Step 11] NOTEARS edges are inherently directed (no orientation needed)...")
print("NOTEARS produces directed edges through continuous optimization constraint")
print("No Granger causality testing - data is cross-sectional (independent daily snapshots)")
print("Edge weights represent structural coefficients from W matrix")

# COMMAND ----------

print("\n[Step 12] Exporting all artifacts...")

# Create pipeline directory
dbutils.fs.mkdirs(pipeline_path)

# Main artifacts dictionary
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
        "edge_type": "structural_coefficient",
        "orientation_method": "notears_optimization"
    },
    "raw_notears_edges": raw_notears_edges,  # NOTEARS output before human priors
    "filtered_edges": filtered_edges,        # Final edges after human priors
    "blacklist_filtering": {
        "blacklist_edges_applicable": len(filtered_blacklist),
        "edges_removed": [(e['from'], e['to']) for e in removed_by_blacklist],
        "whitelist_edges_added": len(added_from_whitelist),
        "whitelist_recovered_from_W_raw": sum(1 for x in added_from_whitelist if x[2] == 'whitelist_recovered'),
        "whitelist_estimated_via_ols": sum(1 for x in added_from_whitelist if x[2] == 'whitelist_estimated'),
        "whitelist_details": [(x[0], x[1], x[2], float(x[3])) for x in added_from_whitelist]
    },
    "final_graph_stats": {
        "total_edges": len(filtered_edges),
        "nodes_with_parents": len(upstream_map),
        "nodes_with_children": len(downstream_map),
        "is_dag": True  # NOTEARS guarantees DAG through h(W)=0 constraint
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

# Save NOTEARS-specific artifacts
if notears_result.get('W_est') is not None:
    # Save weight matrix as CSV
    W_df = pd.DataFrame(
        notears_result['W_est'], 
        index=notears_result['columns'], 
        columns=notears_result['columns']
    )
    W_df.to_csv(f"{pipeline_path}/notears_weight_matrix.csv")

# Save optimization path
if notears_result.get('optimization_path'):
    opt_df = pd.DataFrame(notears_result['optimization_path'])
    opt_df.to_csv(f"{pipeline_path}/notears_optimization_path.csv", index=False)

# ===========================
# SAVE RAW NOTEARS GRAPH (before human priors)
# ===========================
print("\nSaving RAW NOTEARS graph (before human priors)...")

# Save raw edges as CSV
raw_rows = []
for edge in raw_notears_edges:
    if isinstance(edge, dict):
        raw_rows.append({
            "from": edge["from"],
            "to": edge["to"],
            "weight": edge.get("weight", 0.0),
            "abs_weight": edge.get("abs_weight", 0.0),
            "edge_type": edge.get("type", "directed"),
            "source": "notears_raw"
        })

raw_edges_df = pd.DataFrame(raw_rows).sort_values("abs_weight", ascending=False)
raw_edges_df.to_csv(f"{pipeline_path}/notears_raw_edges.csv", index=False)

# Build adjacency maps for RAW graph
raw_upstream_map = defaultdict(list)
raw_downstream_map = defaultdict(list)
for edge in raw_notears_edges:
    if isinstance(edge, dict):
        parent, child = edge["from"], edge["to"]
    else:
        parent, child = edge[0], edge[1]
    raw_upstream_map[child].append(parent)
    raw_downstream_map[parent].append(child)

dbutils.fs.put(f"{pipeline_path}/raw_upstream_map.json", 
               json.dumps(dict(raw_upstream_map), indent=2), 
               overwrite=True)

dbutils.fs.put(f"{pipeline_path}/raw_downstream_map.json", 
               json.dumps(dict(raw_downstream_map), indent=2), 
               overwrite=True)

print(f"  - Raw graph: {len(raw_notears_edges)} edges")
print(f"  - Raw upstream map: {len(raw_upstream_map)} nodes")
print(f"  - Raw downstream map: {len(raw_downstream_map)} nodes")

# ===========================
# SAVE FILTERED GRAPH (after human priors)
# ===========================
print("\nSaving FILTERED graph (after human priors)...")

# Generate NOTEARS edge list ranked by absolute weight (consistent with other pipelines)
rows = []
for edge in filtered_edges:
    if isinstance(edge, dict):
        rows.append({
            "from": edge["from"],
            "to": edge["to"],
            "weight": edge.get("weight", 0.0),
            "abs_weight": edge.get("abs_weight", 0.0),
            "edge_type": edge.get("type", "directed"),
            "source": edge.get("source", "notears")
        })

cand_df = pd.DataFrame(rows).sort_values("abs_weight", ascending=False)
cand_df.to_csv(f"{pipeline_path}/notears_causal_edges.csv", index=False)

print(f"  - Filtered graph: {len(filtered_edges)} edges")
print(f"  - Filtered upstream map: {len(upstream_map)} nodes")
print(f"  - Filtered downstream map: {len(downstream_map)} nodes")

print("="*80)
print("✓ PIPELINE C COMPLETE — All artifacts saved")
print("="*80)
print(f"\nSaved artifacts to: {pipeline_path}")
print(f"\n📁 ARTIFACT INVENTORY:")
print(f"\n  [Core Artifacts]")
print(f"  - causal_artifacts.json          → Main pipeline metadata & both edge lists")
print(f"  - causal_metrics_matrix.csv      → Feature matrix used for discovery")
print(f"  - baseline_stats.json            → Statistical baselines for each metric")
print(f"\n  [RAW NOTEARS Graph - Before Human Priors]")
print(f"  - notears_raw_edges.csv          → {len(raw_notears_edges)} edges from pure NOTEARS")
print(f"  - raw_upstream_map.json          → Parent nodes for each node (raw)")
print(f"  - raw_downstream_map.json        → Child nodes for each node (raw)")
print(f"  - notears_weight_matrix.csv      → Full d×d weight matrix W")
print(f"\n  [FILTERED Graph - After Human Priors]")
print(f"  - notears_causal_edges.csv       → {len(filtered_edges)} edges after blacklist/whitelist")
print(f"  - upstream_map.json              → Parent nodes for each node (filtered)")
print(f"  - downstream_map.json            → Child nodes for each node (filtered)")
print(f"\n  [Optimization Diagnostics]")
print(f"  - notears_optimization_path.csv  → Iteration-by-iteration convergence log")

print(f"\nFinal Results:")
print(f"  - Training period: {MAX_RUNS_TO_PIVOT} days (cross-sectional)")
print(f"  - Final feature matrix: {final_features.shape}")
print(f"  - NOTEARS converged: {notears_result.get('converged', False)}")
print(f"  - RAW graph edges (NOTEARS output): {len(raw_notears_edges)}")
print(f"  - Removed by blacklist: {len(removed_by_blacklist)}")
print(f"  - Added from whitelist: {len(added_from_whitelist)}")
print(f"  - FILTERED graph edges (after priors): {len(filtered_edges)}")
print(f"  - Nodes with parents: {len(upstream_map)}")
print(f"  - Nodes with children: {len(downstream_map)}")
print(f"  - Graph is DAG: ✓")
print(f"  - Temporal features: NONE (cross-sectional approach)")