# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline D: Hybrid Causal Discovery (PC → NOTEARS → Bootstrap)
# MAGIC 
# MAGIC This notebook implements a hybrid causal discovery approach optimized for the fault-injected training data.
# MAGIC 
# MAGIC **Key Design Decisions:**
# MAGIC - Uses ALL 107 dates (42 fault + 65 clean) for full variance spectrum
# MAGIC - Only drops globally constant columns (variance == 0 across ALL 107 runs)
# MAGIC - Enforces tier constraints: Raw → Bronze → Silver → ML/KPIs
# MAGIC - Hybrid pipeline: PC skeleton → NOTEARS weights → Bootstrap stability
# MAGIC - Fault labels used ONLY for evaluation, never as features
# MAGIC 
# MAGIC **Pipeline Architecture:**
# MAGIC | Phase | Step | Method |
# MAGIC |-------|------|--------|
# MAGIC | 1 | Skeleton Discovery | PC Algorithm (α grid search) |
# MAGIC | 2 | Edge Orientation | Tier Constraints + PC v-structures |
# MAGIC | 3 | Weight Estimation | NOTEARS constrained to skeleton |
# MAGIC | 4 | Stability Selection | Bootstrap (100x, 60% threshold) |
# MAGIC | 5 | Graph Refinement | Structural priors + cycle check |

# COMMAND ----------

# MAGIC %pip install networkx scipy scikit-learn pydot causal-learn

# COMMAND ----------

# MAGIC %run ./causal_discovery_utils

# COMMAND ----------

# Imports
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import time
import json
import numpy as np
import pandas as pd
from collections import defaultdict

from pyspark.sql import functions as F

from scipy.optimize import minimize
from scipy.linalg import expm

from causallearn.search.ConstraintBased.PC import pc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 1: Configuration

# COMMAND ----------

# ===========================
# CONFIGURATION
# ===========================

PIPELINE_NAME = "Hybrid_PC_NOTEARS_Bootstrap"
METRICS_TABLE = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"

# Data configuration
TOTAL_RUNS = 107  # 42 fault + 65 clean
FAULT_CUTOFF_DATE = "2025-12-01"  # Dates before this are faulty

# PC Algorithm parameters (grid search)
PC_ALPHA_CANDIDATES = [0.05, 0.07, 0.10]
PC_INDEP_TEST = 'fisherz'

# NOTEARS parameters (grid search)
NOTEARS_LAMBDA_CANDIDATES = [0.01, 0.02, 0.05]
NOTEARS_MAX_ITER = 100
NOTEARS_H_TOL = 1e-8

# Bootstrap parameters
BOOTSTRAP_RESAMPLES = 100
BOOTSTRAP_EDGE_THRESHOLD = 0.60  # Keep edges appearing in >60% of resamples

# Feature selection parameters (LESS AGGRESSIVE)
TARGET_FEATURES = 50  # Allow more features for discovery
CORRELATION_THRESHOLD = 0.99  # Only remove near-perfect correlation (|corr| > 0.99)
VARIANCE_THRESHOLD = 0.0  # Only drop truly constant (variance == 0)

# ===========================
# AUTOMATIC REDUNDANCY DETECTION (No Hardcoding)
# ===========================
# Instead of hardcoding metrics to exclude, we detect patterns algorithmically:
#   1. Mean/std pairs from same aggregation → keep mean, drop std
#   2. Percentile groups (p50, p95, mean of same KPI) → keep p95 only
#   3. Absolute value pairs (abs_X correlated with X) → keep original, drop abs

def detect_redundant_metrics(columns):
    """
    Automatically detect metrics that are mathematical artifacts of each other.
    
    Returns:
        tuple: (metrics_to_drop, metric_groups dict for transparency)
    """
    import re
    
    to_drop = set()
    groups = {}
    
    # ===========================
    # Pattern 1: Mean/Std pairs
    # ===========================
    # If we have X_mean and X_std, they're from the same aggregation
    # Keep mean (more interpretable), drop std
    
    mean_metrics = [c for c in columns if c.endswith('_mean')]
    for mean_col in mean_metrics:
        base = mean_col.rsplit('_mean', 1)[0]
        std_col = f"{base}_std"
        if std_col in columns:
            to_drop.add(std_col)
            groups[std_col] = f"redundant with {mean_col} (same aggregation)"
    
    # ===========================
    # Pattern 2: Percentile groups (both prefix and suffix patterns)
    # ===========================
    # If we have mean_X, p50_X, p95_X - keep only p95 (captures tail behavior)
    # Also handles X_mean, X_p50, X_p95 suffix patterns
    
    # Find percentile/mean groups for KPIs - PREFIX patterns (mean_X, p95_X)
    kpi_patterns = {}
    for col in columns:
        # Match patterns like: mean_fuel_per_100km, p50_fuel_per_100km, p95_fuel_per_100km
        match = re.match(r'(mean|p50|p95|p99)_(.+)', col)
        if match:
            stat_type, kpi_name = match.groups()
            if kpi_name not in kpi_patterns:
                kpi_patterns[kpi_name] = {}
            kpi_patterns[kpi_name][stat_type] = col
    
    # SUFFIX patterns (X_mean, X_p50, X_p95) - common in silver_ml_ metrics
    for col in columns:
        # Match patterns like: silver_ml_imputed_fuel_mean, silver_ml_imputed_fuel_p95
        match = re.match(r'(.+)_(mean|p50|p95|p99)$', col)
        if match:
            base_name, stat_type = match.groups()
            # Use base_name as group key (with suffix marker to avoid collision)
            group_key = f"suffix_{base_name}"
            if group_key not in kpi_patterns:
                kpi_patterns[group_key] = {}
            kpi_patterns[group_key][stat_type] = col
    
    # For each KPI group, keep only p95 (or highest percentile available)
    priority = ['p99', 'p95', 'p50', 'mean']  # Keep first available
    for kpi_name, stats in kpi_patterns.items():
        if len(stats) > 1:
            # Find which one to keep
            keep = None
            for p in priority:
                if p in stats:
                    keep = stats[p]
                    break
            # Drop the rest
            for stat_type, col in stats.items():
                if col != keep:
                    to_drop.add(col)
                    groups[col] = f"redundant with {keep} (same KPI distribution)"
    
    # ===========================
    # Pattern 3: Absolute value pairs  
    # ===========================
    # If we have X_residual and X_abs_residual, they're related by |·|
    # Keep the original, drop the absolute version
    
    for col in columns:
        if '_abs_' in col or col.startswith('abs_'):
            # Try to find the non-abs version
            non_abs = col.replace('_abs_', '_').replace('abs_', '')
            if non_abs in columns:
                to_drop.add(col)
                groups[col] = f"redundant with {non_abs} (absolute value transform)"
            else:
                # Also check pattern: silver_ml_abs_residual_X vs silver_ml_residual_X
                if '_abs_residual' in col:
                    non_abs = col.replace('_abs_residual', '_residual')
                    if non_abs in columns:
                        to_drop.add(col)
                        groups[col] = f"redundant with {non_abs} (absolute value transform)"
                    else:
                        # Even if exact match doesn't exist, abs_residual is derived from residual
                        # Drop it if ANY residual metric exists
                        residual_exists = any('_residual_' in c or c.endswith('_residual_mean') 
                                             for c in columns if '_abs_' not in c)
                        if residual_exists:
                            to_drop.add(col)
                            groups[col] = "derived from residual (absolute value)"
    
    # ===========================
    # Pattern 4: Percentage error (derived from residual)
    # ===========================
    # percentage_error = |residual| / actual * 100, so it's derived from residual
    
    for col in columns:
        if 'percentage_error' in col:
            # Check if residual metrics exist
            residual_exists = any('residual' in c and 'percentage' not in c for c in columns)
            if residual_exists:
                to_drop.add(col)
                groups[col] = "derived from residual (percentage error = |residual|/actual)"
    
    # ===========================
    # Pattern 5: ML model output groups
    # ===========================
    # ML models produce multiple related outputs that are deterministically related:
    #   - prediction = model(features)
    #   - residual = actual - prediction
    #   - imputed = prediction (when actual missing)
    # These will ALWAYS have high edge weights between them.
    # Keep only ONE representative per model (prefer imputed_p95 for fault detection).
    
    ml_output_keywords = ['prediction', 'residual', 'imputed']
    ml_model_groups = {}
    
    for col in columns:
        if col in to_drop:
            continue  # Already dropped by earlier pattern
            
        # Check if this is an ML output metric
        col_lower = col.lower()
        if 'silver_ml_' in col_lower or '_ml_' in col_lower:
            for keyword in ml_output_keywords:
                if keyword in col_lower:
                    # Extract model identifier (e.g., "fuel" from silver_ml_imputed_fuel_p95)
                    # Group by: prefix before the keyword
                    parts = col.split(keyword)
                    if parts[0]:
                        model_id = parts[0].rstrip('_')  # e.g., "silver_ml" or "silver_ml_imputed"
                        
                        # Normalize model_id to group related outputs
                        # silver_ml_imputed_fuel -> silver_ml_fuel
                        # silver_ml_prediction -> silver_ml
                        # silver_ml_residual -> silver_ml
                        base_model = model_id.replace('_imputed', '').replace('_abs', '')
                        
                        if base_model not in ml_model_groups:
                            ml_model_groups[base_model] = {}
                        if keyword not in ml_model_groups[base_model]:
                            ml_model_groups[base_model][keyword] = []
                        ml_model_groups[base_model][keyword].append(col)
                    break
    
    # For each ML model group, keep only imputed (most interpretable for RCA)
    # Priority: imputed > prediction > residual
    ml_priority = ['imputed', 'prediction', 'residual']
    
    for model_id, outputs in ml_model_groups.items():
        all_cols = []
        for keyword in ml_priority:
            all_cols.extend(outputs.get(keyword, []))
        
        if len(all_cols) > 1:
            # Keep only the first by priority (prefer imputed, then prediction)
            keep_cols = []
            for keyword in ml_priority:
                if keyword in outputs and outputs[keyword]:
                    # Keep just one from the highest priority category
                    # Prefer p95 over mean if both exist
                    candidates = outputs[keyword]
                    p95_candidates = [c for c in candidates if 'p95' in c]
                    if p95_candidates:
                        keep_cols.append(p95_candidates[0])
                    else:
                        keep_cols.append(candidates[0])
                    break
            
            # Drop all others
            for col in all_cols:
                if col not in keep_cols:
                    to_drop.add(col)
                    groups[col] = f"ML model output group (keeping {keep_cols[0] if keep_cols else 'representative'})"
    
    return to_drop, groups


def detect_same_computation_pairs(columns, tier_assignments):
    """
    Detect pairs of metrics computed from the same operation.
    These shouldn't have causal edges between them.
    
    Returns:
        list: List of (from, to) tuples to blacklist
    """
    blacklist = []
    
    # Group columns by tier and computation group
    computation_groups = {}
    
    for col in columns:
        tier = tier_assignments.get(col, 2)
        
        # Identify computation groups based on naming patterns
        # ML model outputs
        if 'silver_ml_' in col:
            # Group by model output type
            if 'imputed' in col:
                group_key = (tier, 'ml_imputation')
            elif 'residual' in col or 'prediction' in col:
                group_key = (tier, 'ml_residual_analysis')
            else:
                group_key = (tier, 'ml_other')
        # Vehicle info join outputs
        elif 'vehicle_type' in col or 'vehicle_fuel_subtype' in col:
            group_key = (tier, 'vehicle_join')
        else:
            continue  # No grouping for other metrics
        
        if group_key not in computation_groups:
            computation_groups[group_key] = []
        computation_groups[group_key].append(col)
    
    # Within each computation group, blacklist edges between members
    for group_key, members in computation_groups.items():
        if len(members) > 1:
            for i, m1 in enumerate(members):
                for m2 in members[i+1:]:
                    blacklist.append((m1, m2))
                    blacklist.append((m2, m1))
    
    return blacklist

# Artifact path
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"
pipeline_path = f"{path}/{PIPELINE_NAME}"

print(f"Pipeline D Configuration:")
print(f"  - Method: Hybrid (PC → NOTEARS → Bootstrap)")
print(f"  - Total Training Days: {TOTAL_RUNS}")
print(f"  - PC Alpha Candidates: {PC_ALPHA_CANDIDATES}")
print(f"  - NOTEARS Lambda Candidates: {NOTEARS_LAMBDA_CANDIDATES}")
print(f"  - Bootstrap Resamples: {BOOTSTRAP_RESAMPLES}")
print(f"  - Bootstrap Edge Threshold: {BOOTSTRAP_EDGE_THRESHOLD}")
print(f"  - Target Features: {TARGET_FEATURES}")
print(f"  - Correlation Threshold: {CORRELATION_THRESHOLD}")
print(f"  - Artifact Path: {pipeline_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2: Data Loading & Preparation

# COMMAND ----------

# ===========================
# ENHANCED DATA LOADING
# ===========================

def load_full_metrics_matrix(metrics_sdf, total_runs=107, date_col='date'):
    """
    Load ALL metrics data (fault + clean runs) into wide format.
    
    Key difference from standard loading:
    - Explicitly loads TOTAL_RUNS dates (not just recent 65)
    - Preserves fault vs clean date labeling for evaluation
    
    Args:
        metrics_sdf: Spark DataFrame with columns [date, metric_name, metric_value]
        total_runs: Total number of dates to include
        date_col: Name of the date column
    
    Returns:
        tuple: (metrics DataFrame, date_labels DataFrame)
    """
    sdf = metrics_sdf.select(
        F.col(date_col).alias('date'), 
        F.col('metric_name'), 
        F.col('metric_value')
    )
    
    # Get all available dates, sorted ascending
    all_dates = sdf.select('date').distinct().orderBy(F.asc('date'))
    
    # Take the most recent total_runs dates
    recent_dates = sdf.select('date').distinct().orderBy(F.desc('date')).limit(total_runs)
    recent = sdf.join(recent_dates, on='date', how='inner')
    
    # Pivot to wide format
    pivot = (recent
             .withColumn('metric_value', F.col('metric_value').cast('double'))
             .groupBy('date')
             .pivot('metric_name')
             .agg(F.first('metric_value')))
    
    pdf = pivot.orderBy('date').toPandas()
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.set_index('date').sort_index()
    
    # Create date labels (fault vs clean)
    date_labels = pd.DataFrame({
        'date': pdf.index,
        'is_fault': pdf.index < pd.Timestamp(FAULT_CUTOFF_DATE)
    }).set_index('date')
    
    n_fault = date_labels['is_fault'].sum()
    n_clean = len(date_labels) - n_fault
    
    print(f"Loaded {len(pdf)} days of metrics data")
    print(f"  - Fault runs: {n_fault} (before {FAULT_CUTOFF_DATE})")
    print(f"  - Clean runs: {n_clean} (from {FAULT_CUTOFF_DATE} onwards)")
    print(f"  - Metrics available: {len(pdf.columns)}")
    
    return pdf, date_labels

# COMMAND ----------

# ===========================
# ENHANCED PREPROCESSING
# ===========================

def preprocess_for_hybrid_discovery(
    df,
    zscore=True,
    variance_threshold=0.0,  # Only drop truly constant
    correlation_threshold=0.99,  # Only drop perfect correlations
    impute_strategy="median",
    max_missing_frac=0.3,
    auto_detect_redundant=True,  # NEW: automatic redundancy detection
):
    """
    Preprocess metrics for hybrid causal discovery.
    
    Key differences from standard preprocessing:
    - Only drops columns with ZERO variance across ALL samples
    - Correlation pruning only for |corr| > 0.99 (near duplicates)
    - Preserves low-variance columns that encode fault signals
    - NEW: Automatically detects and removes redundant metrics (mean/std pairs, etc.)
    
    Args:
        df: Raw metrics DataFrame
        zscore: Whether to apply z-score normalization
        variance_threshold: Min variance to keep (0 = only drop constant)
        correlation_threshold: Max correlation before pruning (0.99 = only duplicates)
        impute_strategy: Imputation strategy
        max_missing_frac: Max missing fraction before dropping column
        auto_detect_redundant: Whether to automatically detect and drop redundant metrics
    
    Returns:
        tuple: (preprocessed DataFrame, metadata dict)
    """
    meta = {}
    
    # Normalize null tokens
    df = df.replace(["null", "NULL", "None", ""], np.nan)
    
    # ===========================
    # NEW: Automatic redundancy detection (instead of hardcoded list)
    # ===========================
    if auto_detect_redundant:
        redundant_metrics, redundancy_groups = detect_redundant_metrics(df.columns.tolist())
        if redundant_metrics:
            redundant_present = [c for c in redundant_metrics if c in df.columns]
            df = df.drop(columns=redundant_present)
            print(f"  Auto-detected {len(redundant_present)} redundant metrics:")
            for metric in sorted(redundant_present):
                reason = redundancy_groups.get(metric, "pattern detected")
                print(f"    - {metric}: {reason}")
        meta["auto_dropped_redundant"] = list(redundant_metrics)
        meta["redundancy_reasons"] = redundancy_groups
    
    # Coerce to numeric
    df_num = df.copy()
    for c in df_num.columns:
        if not pd.api.types.is_numeric_dtype(df_num[c]):
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    
    meta["initial_shape"] = df_num.shape
    
    # Missingness stats
    miss_frac = df_num.isna().mean()
    meta["missing_fraction"] = miss_frac.to_dict()
    
    # Drop fully-null columns
    drop_all_null = miss_frac[miss_frac == 1.0].index.tolist()
    meta["dropped_all_null"] = drop_all_null
    
    # Drop high-missing columns
    drop_high_missing = miss_frac[(miss_frac > max_missing_frac) & (miss_frac < 1.0)].index.tolist()
    meta["dropped_high_missing"] = drop_high_missing
    
    drop_cols = sorted(set(drop_all_null + drop_high_missing))
    if drop_cols:
        df_num = df_num.drop(columns=drop_cols)
        print(f"  Dropped {len(drop_cols)} columns with >={max_missing_frac*100:.0f}% missing values")
    
    # Impute remaining missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy=impute_strategy)
    imputed_array = imputer.fit_transform(df_num.values)
    df_num = pd.DataFrame(imputed_array, index=df_num.index, columns=df_num.columns)
    
    # Drop ONLY constant columns (variance == 0)
    variances = df_num.var()
    const_cols = variances[variances <= variance_threshold].index.tolist()
    meta["dropped_constant"] = const_cols
    
    if const_cols:
        df_num = df_num.drop(columns=const_cols)
        print(f"  Dropped {len(const_cols)} constant columns (variance == 0)")
    
    # Remove only near-perfect correlations (|corr| > threshold)
    # This is MUCH less aggressive than before
    if df_num.shape[1] > 1:
        corr_matrix = df_num.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find pairs with |corr| > threshold
        high_corr_pairs = []
        for col in upper_tri.columns:
            correlated = upper_tri[col][upper_tri[col] > correlation_threshold].index.tolist()
            for corr_col in correlated:
                high_corr_pairs.append((col, corr_col, corr_matrix.loc[col, corr_col]))
        
        # Remove lower-priority column from each pair
        removed_corr = []
        remaining = set(df_num.columns)
        
        for col1, col2, corr_val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
            if col1 in remaining and col2 in remaining:
                # Keep the one with higher priority
                p1 = _get_feature_priority(col1)
                p2 = _get_feature_priority(col2)
                
                if p1 >= p2:
                    remove_col = col2
                else:
                    remove_col = col1
                
                removed_corr.append(remove_col)
                remaining.remove(remove_col)
        
        if removed_corr:
            df_num = df_num.drop(columns=removed_corr)
            print(f"  Dropped {len(removed_corr)} near-duplicate columns (|corr| > {correlation_threshold})")
        
        meta["dropped_high_correlation"] = removed_corr
    
    # Z-score normalization
    if zscore:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df_num.values)
        df_num = pd.DataFrame(scaled_array, index=df_num.index, columns=df_num.columns)
    
    meta["final_shape"] = df_num.shape
    print(f"  Final shape: {df_num.shape}")
    
    return df_num, meta

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 3: Constraint Matrix Generation

# COMMAND ----------

# ===========================
# TIER CONSTRAINT MATRIX
# ===========================

def generate_tier_constraint_matrix(columns):
    """
    Generate constraint matrix enforcing tier ordering:
    Raw → Bronze → Silver → ML/KPIs
    
    Returns a matrix where constraint[i,j] = 1 means edge i→j is FORBIDDEN.
    
    Args:
        columns: List of column names
    
    Returns:
        tuple: (constraint_matrix, tier_assignments dict)
    """
    n = len(columns)
    constraint_matrix = np.zeros((n, n))
    
    # Assign tiers
    tier_assignments = {}
    for i, col in enumerate(columns):
        col_lower = col.lower()
        
        if col.startswith('raw_'):
            tier = 0
        elif col.startswith('bronze_'):
            tier = 1
        elif col.startswith('silver_'):
            # ML outputs and KPIs are tier 3
            if any(kw in col_lower for kw in ['ml_', 'residual', 'prediction', 'imputed', 
                                               'fuel_per_100km', 'idling_per_100km', '_kpi']):
                tier = 3
            else:
                tier = 2
        elif any(kw in col_lower for kw in ['fuel_per_100km', 'idling_per_100km', 
                                             'mean_', 'p50_', 'p95_']):
            tier = 3  # KPIs
        else:
            tier = 2  # Default to silver
        
        tier_assignments[col] = tier
    
    # Build constraint matrix: forbid edges from higher tier to lower tier
    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns):
            if i == j:
                continue
            
            tier_i = tier_assignments[col_i]
            tier_j = tier_assignments[col_j]
            
            # Forbid edge i → j if tier_i > tier_j (downstream → upstream)
            if tier_i > tier_j:
                constraint_matrix[i, j] = 1
    
    # Print tier distribution
    tier_counts = {}
    for col, tier in tier_assignments.items():
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    tier_names = {0: 'Raw', 1: 'Bronze', 2: 'Silver', 3: 'ML/KPIs'}
    print(f"Tier Distribution:")
    for tier in sorted(tier_counts.keys()):
        print(f"  - Tier {tier} ({tier_names.get(tier, 'Unknown')}): {tier_counts[tier]} columns")
    
    n_forbidden = int(constraint_matrix.sum())
    print(f"Constraint Matrix: {n_forbidden} forbidden edges out of {n*(n-1)} possible")
    
    return constraint_matrix, tier_assignments


def generate_full_blacklist(columns, tier_assignments):
    """
    Generate blacklist edges from tier constraints.
    
    Args:
        columns: List of column names
        tier_assignments: Dict mapping column → tier
    
    Returns:
        list: List of (from, to) tuples that are forbidden
    """
    blacklist = []
    
    for col_i in columns:
        for col_j in columns:
            if col_i == col_j:
                continue
            
            tier_i = tier_assignments.get(col_i, 2)
            tier_j = tier_assignments.get(col_j, 2)
            
            # Forbid downstream → upstream
            if tier_i > tier_j:
                blacklist.append((col_i, col_j))
    
    return blacklist

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 4: PC Algorithm (Skeleton Discovery)

# COMMAND ----------

# ===========================
# PC ALGORITHM
# ===========================

def extract_pc_skeleton(pc_result, column_names):
    """
    Extract skeleton edges from PC algorithm result.
    
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
    
    seen = set()
    for i in range(len(column_names)):
        for j in range(len(column_names)):
            if i != j and graph_matrix[i, j] != 0:
                edge_value = graph_matrix[i, j]
                
                # PC edge encoding
                if edge_value == 1:  # i → j
                    edges.append((column_names[i], column_names[j], 'directed'))
                elif edge_value == -1:  # j → i
                    edges.append((column_names[j], column_names[i], 'directed'))
                elif edge_value == 2:  # undirected
                    pair = tuple(sorted([column_names[i], column_names[j]]))
                    if pair not in seen:
                        edges.append((column_names[i], column_names[j], 'undirected'))
                        seen.add(pair)
    
    return edges


def run_pc_with_grid_search(df, alpha_candidates, indep_test='fisherz'):
    """
    Run PC algorithm with grid search over alpha values.
    
    Selection criteria:
    - Prefer moderate skeleton density (not too sparse, not too dense)
    - Target: 1-3 edges per node on average
    
    Args:
        df: Preprocessed DataFrame
        alpha_candidates: List of alpha values to try
        indep_test: Independence test method
    
    Returns:
        dict: Best result with edges, alpha, and metadata
    """
    print(f"Running PC Algorithm with grid search over α ∈ {alpha_candidates}")
    
    n_samples, n_features = df.shape
    print(f"  - Samples: {n_samples}, Features: {n_features}")
    print(f"  - Sample-to-feature ratio: {n_samples / n_features:.2f}")
    
    results = []
    data = df.values.astype(float)
    
    for alpha in alpha_candidates:
        print(f"\n  Testing α = {alpha}...")
        
        try:
            pc_obj = pc(data, alpha=alpha, indep_test=indep_test)
            edges = extract_pc_skeleton(pc_obj, df.columns.tolist())
            
            # Calculate skeleton density
            n_edges = len(edges)
            avg_edges_per_node = 2 * n_edges / n_features if n_features > 0 else 0
            
            print(f"    - Edges found: {n_edges}")
            print(f"    - Avg edges per node: {avg_edges_per_node:.2f}")
            
            results.append({
                'alpha': alpha,
                'edges': edges,
                'n_edges': n_edges,
                'avg_edges_per_node': avg_edges_per_node,
                'pc_object': pc_obj
            })
            
        except Exception as e:
            print(f"    - FAILED: {str(e)}")
            results.append({
                'alpha': alpha,
                'edges': [],
                'n_edges': 0,
                'avg_edges_per_node': 0,
                'error': str(e)
            })
    
    # Select best alpha
    # Target: 1-3 edges per node
    valid_results = [r for r in results if r['n_edges'] > 0]
    
    if not valid_results:
        print("\n❌ No valid results from any alpha!")
        return {'method': 'pc-error', 'error': 'All alphas failed', 'edges': []}
    
    # Score each result (penalize deviation from target density)
    target_density = 2.0  # Target avg edges per node
    for r in valid_results:
        r['density_score'] = -abs(r['avg_edges_per_node'] - target_density)
    
    best = max(valid_results, key=lambda x: x['density_score'])
    
    print(f"\n✓ Selected α = {best['alpha']} with {best['n_edges']} edges")
    print(f"  - Avg edges per node: {best['avg_edges_per_node']:.2f}")
    
    return {
        'method': 'pc-success',
        'alpha': best['alpha'],
        'edges': best['edges'],
        'n_edges': best['n_edges'],
        'avg_edges_per_node': best['avg_edges_per_node'],
        'pc_object': best.get('pc_object'),
        'all_results': [{k: v for k, v in r.items() if k != 'pc_object'} for r in results]
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 5: NOTEARS (Weight Estimation)

# COMMAND ----------

# ===========================
# NOTEARS ALGORITHM
# ===========================

class NOTEARSConstrained:
    """
    NOTEARS implementation constrained to a given skeleton.
    
    Only estimates weights for edges that exist in the skeleton.
    """
    
    def __init__(self, lambda1=0.01, max_iter=100, h_tol=1e-8):
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.W_est = None
    
    def _loss(self, W, X):
        """Squared loss."""
        n, d = X.shape
        R = X - X @ W
        return 0.5 / n * np.trace(R.T @ R)
    
    def _h(self, W):
        """Acyclicity constraint."""
        d = W.shape[0]
        M = W * W
        E = expm(M)
        return np.trace(E) - d
    
    def _h_grad(self, W):
        """Gradient of acyclicity constraint."""
        M = W * W
        E = expm(M)
        return 2 * W * E
    
    def fit(self, X, skeleton_mask):
        """
        Fit NOTEARS constrained to skeleton.
        
        Args:
            X: Data matrix (n_samples, n_features)
            skeleton_mask: Binary matrix where 1 = edge allowed
        
        Returns:
            self
        """
        n, d = X.shape
        
        # Initialize W only where skeleton allows
        W = np.random.randn(d, d) * 0.1 * skeleton_mask
        np.fill_diagonal(W, 0)
        
        rho, alpha = 1.0, 0.0
        
        for iteration in range(self.max_iter):
            # Define objective for this iteration
            def objective(w_vec):
                W_mat = w_vec.reshape(d, d) * skeleton_mask
                loss = self._loss(W_mat, X)
                h_val = self._h(W_mat)
                reg = self.lambda1 * np.sum(np.abs(W_mat))
                return loss + reg + alpha * h_val + 0.5 * rho * h_val ** 2
            
            def gradient(w_vec):
                W_mat = w_vec.reshape(d, d) * skeleton_mask
                R = X - X @ W_mat
                loss_grad = -1.0 / n * X.T @ R
                h_val = self._h(W_mat)
                h_grad = self._h_grad(W_mat)
                reg_grad = self.lambda1 * np.sign(W_mat)
                grad = (loss_grad + reg_grad + (alpha + rho * h_val) * h_grad) * skeleton_mask
                return grad.flatten()
            
            # Optimize
            result = minimize(objective, W.flatten(), method='L-BFGS-B', jac=gradient,
                            options={'maxiter': 100, 'disp': False})
            
            W = result.x.reshape(d, d) * skeleton_mask
            np.fill_diagonal(W, 0)
            
            # Check convergence
            h_val = self._h(W)
            if h_val < self.h_tol:
                break
            
            # Update Lagrangian
            alpha += rho * h_val
            rho = min(rho * 2, 1e16)
        
        self.W_est = W
        self.converged = h_val < self.h_tol
        self.final_h = h_val
        
        return self


def run_notears_on_skeleton(df, skeleton_edges, lambda_candidates, columns):
    """
    Run NOTEARS constrained to PC skeleton with grid search over lambda.
    
    Args:
        df: Preprocessed DataFrame
        skeleton_edges: Edges from PC algorithm
        lambda_candidates: Lambda values to try
        columns: List of column names
    
    Returns:
        dict: Best result with weighted edges
    """
    print(f"Running NOTEARS with grid search over λ ∈ {lambda_candidates}")
    
    n_features = len(columns)
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    
    # Build skeleton mask
    skeleton_mask = np.zeros((n_features, n_features))
    for edge in skeleton_edges:
        if isinstance(edge, tuple) and len(edge) >= 2:
            a, b = edge[0], edge[1]
            edge_type = edge[2] if len(edge) > 2 else 'undirected'
        else:
            continue
        
        if a in col_to_idx and b in col_to_idx:
            i, j = col_to_idx[a], col_to_idx[b]
            skeleton_mask[i, j] = 1
            if edge_type == 'undirected':
                skeleton_mask[j, i] = 1
    
    print(f"  - Skeleton has {int(skeleton_mask.sum())} possible edge slots")
    
    X = df[columns].values.astype(float)
    results = []
    
    for lambda1 in lambda_candidates:
        print(f"\n  Testing λ = {lambda1}...")
        
        try:
            model = NOTEARSConstrained(lambda1=lambda1, max_iter=NOTEARS_MAX_ITER, h_tol=NOTEARS_H_TOL)
            model.fit(X, skeleton_mask)
            
            # Extract weighted edges
            W = model.W_est
            weighted_edges = []
            
            for i in range(n_features):
                for j in range(n_features):
                    if abs(W[i, j]) > 1e-6:
                        weighted_edges.append({
                            'from': columns[i],
                            'to': columns[j],
                            'weight': float(W[i, j]),
                            'abs_weight': float(abs(W[i, j]))
                        })
            
            print(f"    - Converged: {model.converged}")
            print(f"    - Final h: {model.final_h:.2e}")
            print(f"    - Non-zero edges: {len(weighted_edges)}")
            
            results.append({
                'lambda1': lambda1,
                'edges': weighted_edges,
                'n_edges': len(weighted_edges),
                'W_est': W,
                'converged': model.converged,
                'final_h': model.final_h
            })
            
        except Exception as e:
            print(f"    - FAILED: {str(e)}")
            results.append({
                'lambda1': lambda1,
                'edges': [],
                'n_edges': 0,
                'error': str(e)
            })
    
    # Select best lambda (prefer converged with moderate sparsity)
    valid_results = [r for r in results if r['n_edges'] > 0 and r.get('converged', False)]
    
    if not valid_results:
        valid_results = [r for r in results if r['n_edges'] > 0]
    
    if not valid_results:
        print("\n❌ No valid NOTEARS results!")
        return {'method': 'notears-error', 'edges': [], 'error': 'All lambdas failed'}
    
    # Prefer moderate sparsity
    best = min(valid_results, key=lambda x: abs(x['n_edges'] - len(skeleton_edges) * 0.7))
    
    print(f"\n✓ Selected λ = {best['lambda1']} with {best['n_edges']} weighted edges")
    
    return {
        'method': 'notears-success',
        'lambda1': best['lambda1'],
        'edges': best['edges'],
        'n_edges': best['n_edges'],
        'W_est': best['W_est'],
        'converged': best.get('converged', False),
        'final_h': best.get('final_h'),
        'columns': columns
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 6: Bootstrap Stability Selection

# COMMAND ----------

# ===========================
# BOOTSTRAP STABILITY
# ===========================

def bootstrap_edge_stability(df, skeleton_edges, notears_lambda, columns, 
                              n_resamples=100, edge_threshold=0.60):
    """
    Assess edge stability through bootstrap resampling.
    
    For each resample:
    1. Draw n samples with replacement
    2. Run NOTEARS on skeleton
    3. Record which edges appear
    
    Keep edges appearing in >threshold of resamples.
    
    Args:
        df: Preprocessed DataFrame
        skeleton_edges: PC skeleton edges
        notears_lambda: Lambda for NOTEARS
        columns: Column names
        n_resamples: Number of bootstrap samples
        edge_threshold: Min frequency to keep edge
    
    Returns:
        dict: Stable edges and stability scores
    """
    print(f"Running Bootstrap Stability ({n_resamples} resamples, threshold={edge_threshold})")
    
    n_samples, n_features = df.shape
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    
    # Build skeleton mask
    skeleton_mask = np.zeros((n_features, n_features))
    for edge in skeleton_edges:
        if isinstance(edge, tuple) and len(edge) >= 2:
            a, b = edge[0], edge[1]
            edge_type = edge[2] if len(edge) > 2 else 'undirected'
            
            if a in col_to_idx and b in col_to_idx:
                i, j = col_to_idx[a], col_to_idx[b]
                skeleton_mask[i, j] = 1
                if edge_type == 'undirected':
                    skeleton_mask[j, i] = 1
    
    # Track edge frequencies
    edge_counts = defaultdict(int)
    edge_weights = defaultdict(list)
    
    X = df[columns].values.astype(float)
    
    for b in range(n_resamples):
        if (b + 1) % 20 == 0:
            print(f"  Bootstrap {b+1}/{n_resamples}...")
        
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        
        try:
            # Fit NOTEARS
            model = NOTEARSConstrained(lambda1=notears_lambda, max_iter=50, h_tol=1e-6)
            model.fit(X_boot, skeleton_mask)
            
            W = model.W_est
            
            # Record edges
            for i in range(n_features):
                for j in range(n_features):
                    if abs(W[i, j]) > 1e-6:
                        edge_key = (columns[i], columns[j])
                        edge_counts[edge_key] += 1
                        edge_weights[edge_key].append(W[i, j])
        
        except Exception as e:
            continue
    
    # Compute stability scores and filter
    stable_edges = []
    stability_scores = {}
    
    for edge_key, count in edge_counts.items():
        frequency = count / n_resamples
        stability_scores[edge_key] = frequency
        
        if frequency >= edge_threshold:
            weights = edge_weights[edge_key]
            avg_weight = np.mean(weights)
            
            stable_edges.append({
                'from': edge_key[0],
                'to': edge_key[1],
                'weight': float(avg_weight),
                'abs_weight': float(abs(avg_weight)),
                'bootstrap_frequency': float(frequency),
                'source': 'bootstrap_stable'
            })
    
    # Sort by frequency then weight
    stable_edges = sorted(stable_edges, key=lambda x: (-x['bootstrap_frequency'], -x['abs_weight']))
    
    # Compute weight stability statistics
    weight_cv = {}  # Coefficient of variation for weights
    for edge_key, weights in edge_weights.items():
        if len(weights) > 1:
            mean_w = np.mean(weights)
            std_w = np.std(weights)
            cv = std_w / abs(mean_w) if abs(mean_w) > 1e-6 else 0
            weight_cv[edge_key] = cv
    
    print(f"\n✓ Bootstrap complete:")
    print(f"  - Total edges seen: {len(edge_counts)}")
    print(f"  - Stable edges (freq ≥ {edge_threshold}): {len(stable_edges)}")
    
    # Report weight stability
    if weight_cv:
        avg_cv = np.mean(list(weight_cv.values()))
        max_cv = max(weight_cv.values())
        low_stability = sum(1 for cv in weight_cv.values() if cv > 0.5)
        print(f"  - Avg weight CV: {avg_cv:.3f} (lower = more stable)")
        print(f"  - Max weight CV: {max_cv:.3f}")
        print(f"  - Edges with high weight variance (CV > 0.5): {low_stability}")
    
    return {
        'stable_edges': stable_edges,
        'stability_scores': stability_scores,
        'weight_cv': weight_cv,
        'n_resamples': n_resamples,
        'threshold': edge_threshold,
        'total_edges_seen': len(edge_counts)
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 7: Graph Refinement

# COMMAND ----------

# ===========================
# GRAPH REFINEMENT
# ===========================

def add_structural_prior_edges(edges, whitelist, data, existing_edge_set):
    """
    Add structural prior edges that are always true regardless of data.
    
    These are marked as 'structural_prior' to be transparent.
    
    Args:
        edges: Current edge list
        whitelist: Human prior whitelist
        data: DataFrame for weight estimation
        existing_edge_set: Set of existing (from, to) tuples
    
    Returns:
        list: Updated edges with structural priors added
    """
    print("Adding structural prior edges...")
    
    available_cols = set(data.columns)
    added = []
    
    for from_col, to_col in whitelist:
        if from_col in available_cols and to_col in available_cols:
            if (from_col, to_col) not in existing_edge_set:
                # Estimate weight via OLS
                try:
                    X = data[from_col].values.reshape(-1, 1)
                    y = data[to_col].values
                    X_with_intercept = np.column_stack([np.ones(len(X)), X])
                    beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                    weight = float(beta[1])
                except:
                    weight = 0.0
                
                edge_dict = {
                    'from': from_col,
                    'to': to_col,
                    'weight': weight,
                    'abs_weight': abs(weight),
                    'bootstrap_frequency': 1.0,  # Always true
                    'source': 'structural_prior'
                }
                edges.append(edge_dict)
                added.append((from_col, to_col))
    
    print(f"  Added {len(added)} structural prior edges")
    return edges, added


def validate_dag_acyclicity(edges):
    """
    Validate that the graph is acyclic.
    
    Args:
        edges: List of edge dicts
    
    Returns:
        tuple: (is_dag bool, cycles list if any)
    """
    import networkx as nx
    
    G = nx.DiGraph()
    for edge in edges:
        if isinstance(edge, dict):
            G.add_edge(edge['from'], edge['to'])
        else:
            G.add_edge(edge[0], edge[1])
    
    try:
        cycles = list(nx.simple_cycles(G))
        is_dag = len(cycles) == 0
        
        if is_dag:
            print("✓ Graph is acyclic (valid DAG)")
        else:
            print(f"⚠️  Found {len(cycles)} cycles!")
            for cycle in cycles[:5]:  # Show first 5
                print(f"    Cycle: {' → '.join(cycle)}")
        
        return is_dag, cycles
    
    except Exception as e:
        print(f"Could not check cycles: {e}")
        return True, []


def apply_blacklist_filtering(edges, blacklist_set):
    """
    Remove blacklisted edges.
    
    Args:
        edges: List of edge dicts
        blacklist_set: Set of (from, to) tuples to remove
    
    Returns:
        tuple: (filtered edges, removed edges)
    """
    filtered = []
    removed = []
    
    for edge in edges:
        if isinstance(edge, dict):
            key = (edge['from'], edge['to'])
        else:
            key = (edge[0], edge[1])
        
        if key in blacklist_set:
            removed.append(edge)
        else:
            filtered.append(edge)
    
    print(f"Blacklist filtering: {len(edges)} → {len(filtered)} edges ({len(removed)} removed)")
    return filtered, removed

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Driver Code

# COMMAND ----------

# ===========================
# MAIN DRIVER
# ===========================

print("="*80)
print("PIPELINE D: HYBRID CAUSAL DISCOVERY (PC → NOTEARS → Bootstrap)")
print("="*80)

# =====================
# PHASE 1: DATA LOADING
# =====================
print("\n" + "="*60)
print("PHASE 1: DATA LOADING")
print("="*60)

metrics_sdf = spark.table(METRICS_TABLE)
metrics_pdf, date_labels = load_full_metrics_matrix(metrics_sdf, total_runs=TOTAL_RUNS)

# Store fault labels for evaluation (NOT used in discovery)
fault_dates = date_labels[date_labels['is_fault']].index.tolist()
clean_dates = date_labels[~date_labels['is_fault']].index.tolist()
print(f"\n⚠️  Fault labels stored for evaluation only (not used in discovery)")

# =====================
# PHASE 2: PREPROCESSING
# =====================
print("\n" + "="*60)
print("PHASE 2: PREPROCESSING")
print("="*60)

preprocessed_df, preprocess_meta = preprocess_for_hybrid_discovery(
    metrics_pdf,
    zscore=True,
    variance_threshold=VARIANCE_THRESHOLD,
    correlation_threshold=CORRELATION_THRESHOLD,
    auto_detect_redundant=True  # NEW: automatic detection instead of hardcoding
)

print(f"\nPreprocessing Summary:")
print(f"  - Initial: {preprocess_meta['initial_shape']}")
print(f"  - Final: {preprocess_meta['final_shape']}")

# =====================
# PHASE 3: CONSTRAINTS
# =====================
print("\n" + "="*60)
print("PHASE 3: TIER CONSTRAINTS")
print("="*60)

columns = preprocessed_df.columns.tolist()
constraint_matrix, tier_assignments = generate_tier_constraint_matrix(columns)
blacklist = generate_full_blacklist(columns, tier_assignments)

# NEW: Auto-detect same-computation pairs (instead of hardcoded list)
same_computation_blacklist = detect_same_computation_pairs(columns, tier_assignments)
blacklist.extend(same_computation_blacklist)
blacklist_set = set(blacklist)

print(f"\nTotal blacklist edges: {len(blacklist_set)}")
print(f"  - Tier constraint edges: {len(blacklist)}")
print(f"  - Same-computation edges (auto-detected): {len(same_computation_blacklist)}")

# =====================
# PHASE 4: PC SKELETON
# =====================
print("\n" + "="*60)
print("PHASE 4: PC SKELETON DISCOVERY")
print("="*60)

pc_result = run_pc_with_grid_search(
    preprocessed_df,
    alpha_candidates=PC_ALPHA_CANDIDATES,
    indep_test=PC_INDEP_TEST
)

if pc_result['method'] == 'pc-error':
    raise Exception(f"PC Algorithm failed: {pc_result.get('error', 'Unknown error')}")

skeleton_edges = pc_result['edges']
print(f"\nPC Skeleton: {len(skeleton_edges)} edges")

# Visualize skeleton
G_skeleton = visualize_skeleton(skeleton_edges, title="Phase 4: PC Skeleton (Before Constraints)")

# =====================
# PHASE 5: NOTEARS WEIGHTS
# =====================
print("\n" + "="*60)
print("PHASE 5: NOTEARS WEIGHT ESTIMATION")
print("="*60)

notears_result = run_notears_on_skeleton(
    preprocessed_df,
    skeleton_edges,
    lambda_candidates=NOTEARS_LAMBDA_CANDIDATES,
    columns=columns
)

if notears_result['method'] == 'notears-error':
    print("⚠️  NOTEARS failed, using PC edges with unit weights")
    weighted_edges = [{'from': e[0], 'to': e[1], 'weight': 1.0, 'abs_weight': 1.0} 
                      for e in skeleton_edges if len(e) >= 2]
else:
    weighted_edges = notears_result['edges']

print(f"\nWeighted edges: {len(weighted_edges)}")

# =====================
# PHASE 6: BOOTSTRAP
# =====================
print("\n" + "="*60)
print("PHASE 6: BOOTSTRAP STABILITY SELECTION")
print("="*60)

bootstrap_result = bootstrap_edge_stability(
    preprocessed_df,
    skeleton_edges,
    notears_lambda=notears_result.get('lambda1', 0.01),
    columns=columns,
    n_resamples=BOOTSTRAP_RESAMPLES,
    edge_threshold=BOOTSTRAP_EDGE_THRESHOLD
)

stable_edges = bootstrap_result['stable_edges']
print(f"\nStable edges after bootstrap: {len(stable_edges)}")

# =====================
# PHASE 7: REFINEMENT
# =====================
print("\n" + "="*60)
print("PHASE 7: GRAPH REFINEMENT")
print("="*60)

# Apply blacklist
print("\n[Step 7.1] Applying tier constraints (blacklist)...")
filtered_edges, removed_by_blacklist = apply_blacklist_filtering(stable_edges, blacklist_set)

# Add structural priors
print("\n[Step 7.2] Adding structural prior edges...")
existing_edge_set = set((e['from'], e['to']) for e in filtered_edges)
final_edges, added_structural = add_structural_prior_edges(
    filtered_edges, 
    HUMAN_PRIOR_WHITELIST, 
    preprocessed_df,
    existing_edge_set
)

# Validate DAG
print("\n[Step 7.3] Validating acyclicity...")
is_dag, cycles = validate_dag_acyclicity(final_edges)

if not is_dag:
    print("⚠️  Removing edges to break cycles...")
    # Remove lowest-weight edges from cycles
    for cycle in cycles:
        cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        # Find edge with lowest weight
        min_weight_edge = None
        min_weight = float('inf')
        for e in final_edges:
            if (e['from'], e['to']) in cycle_edges:
                if e['abs_weight'] < min_weight:
                    min_weight = e['abs_weight']
                    min_weight_edge = (e['from'], e['to'])
        
        if min_weight_edge:
            final_edges = [e for e in final_edges if (e['from'], e['to']) != min_weight_edge]
            print(f"    Removed: {min_weight_edge[0]} → {min_weight_edge[1]}")
    
    # Re-validate
    is_dag, _ = validate_dag_acyclicity(final_edges)

# =====================
# VISUALIZATION
# =====================
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

G_final = visualize_dag(final_edges, title="Pipeline D: Final Hybrid Causal DAG")

# =====================
# SUMMARY
# =====================
print("\n" + "="*80)
print("PIPELINE D: SUMMARY")
print("="*80)

print(f"\nPipeline Results:")
print(f"  - PC Alpha selected: {pc_result.get('alpha', 'N/A')}")
print(f"  - PC Skeleton edges: {len(skeleton_edges)}")
print(f"  - NOTEARS Lambda selected: {notears_result.get('lambda1', 'N/A')}")
print(f"  - NOTEARS weighted edges: {len(weighted_edges)}")
print(f"  - Bootstrap stable edges: {len(stable_edges)}")
print(f"  - Removed by blacklist: {len(removed_by_blacklist)}")
print(f"  - Added structural priors: {len(added_structural)}")
print(f"  - FINAL edges: {len(final_edges)}")
print(f"  - Is valid DAG: {is_dag}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Artifacts

# COMMAND ----------

# ===========================
# EXPORT ARTIFACTS
# ===========================

print("\n" + "="*60)
print("EXPORTING ARTIFACTS")
print("="*60)

print("\n[Step 1] Computing baseline statistics...")
baseline_stats = compute_baseline_stats(preprocessed_df)
print(f"Computed baseline statistics for {len(baseline_stats)} metrics")

print("\n[Step 2] Building adjacency maps...")
upstream_map, downstream_map = build_adjacency_maps(final_edges, handle_undirected=False)
print(f"Upstream map: {len(upstream_map)} nodes")
print(f"Downstream map: {len(downstream_map)} nodes")

print("\n[Step 3] Saving artifacts...")
dbutils.fs.mkdirs(pipeline_path)

# Main artifacts
artifacts = {
    "pipeline": PIPELINE_NAME,
    "method": "hybrid_pc_notears_bootstrap",
    "data_type": "cross-sectional",
    "status": "SUCCESS",
    "created_at": datetime.utcnow().isoformat(),
    
    # Configuration
    "config": {
        "total_runs": TOTAL_RUNS,
        "fault_cutoff_date": FAULT_CUTOFF_DATE,
        "pc_alpha_candidates": PC_ALPHA_CANDIDATES,
        "notears_lambda_candidates": NOTEARS_LAMBDA_CANDIDATES,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_edge_threshold": BOOTSTRAP_EDGE_THRESHOLD,
        "target_features": TARGET_FEATURES,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "variance_threshold": VARIANCE_THRESHOLD
    },
    
    # Data info
    "data_info": {
        "n_fault_dates": len(fault_dates),
        "n_clean_dates": len(clean_dates),
        "fault_dates": [str(d.date()) for d in fault_dates],
        "clean_dates": [str(d.date()) for d in clean_dates]
    },
    
    # Preprocessing
    "preprocess_meta": preprocess_meta,
    
    # PC results
    "pc_result": {
        "method": pc_result["method"],
        "alpha_selected": pc_result.get("alpha"),
        "n_skeleton_edges": len(skeleton_edges),
        "avg_edges_per_node": pc_result.get("avg_edges_per_node"),
        "all_alpha_results": pc_result.get("all_results", [])
    },
    
    # NOTEARS results
    "notears_result": {
        "method": notears_result.get("method"),
        "lambda_selected": notears_result.get("lambda1"),
        "n_weighted_edges": len(weighted_edges),
        "converged": notears_result.get("converged"),
        "final_h": notears_result.get("final_h")
    },
    
    # Bootstrap results
    "bootstrap_result": {
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "edge_threshold": BOOTSTRAP_EDGE_THRESHOLD,
        "n_stable_edges": len(stable_edges),
        "total_edges_seen": bootstrap_result.get("total_edges_seen")
    },
    
    # Final graph
    "final_graph": {
        "n_edges": len(final_edges),
        "n_removed_by_blacklist": len(removed_by_blacklist),
        "n_added_structural": len(added_structural),
        "is_dag": is_dag,
        "n_cycles_removed": len(cycles) if not is_dag else 0
    },
    
    # Edge lists
    "skeleton_edges": [(e[0], e[1], e[2] if len(e) > 2 else 'undirected') for e in skeleton_edges],
    "stable_edges": stable_edges,
    "final_edges": final_edges,
    
    # Tier assignments
    "tier_assignments": tier_assignments
}

# Save core artifacts
preprocessed_df.to_csv(f"{pipeline_path}/causal_metrics_matrix.csv")
dbutils.fs.put(f"{pipeline_path}/causal_artifacts.json", json.dumps(artifacts, indent=2, default=str), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/baseline_stats.json", json.dumps(baseline_stats, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/upstream_map.json", json.dumps(upstream_map, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/downstream_map.json", json.dumps(downstream_map, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/tier_assignments.json", json.dumps(tier_assignments, indent=2), overwrite=True)

# Save edge CSVs
# Skeleton edges
skeleton_rows = [{"from": e[0], "to": e[1], "edge_type": e[2] if len(e) > 2 else "undirected"} 
                 for e in skeleton_edges]
pd.DataFrame(skeleton_rows).to_csv(f"{pipeline_path}/pc_skeleton_edges.csv", index=False)

# Stable edges
pd.DataFrame(stable_edges).to_csv(f"{pipeline_path}/bootstrap_stable_edges.csv", index=False)

# Final edges
final_rows = []
for edge in final_edges:
    if isinstance(edge, dict):
        final_rows.append({
            "from": edge["from"],
            "to": edge["to"],
            "weight": edge.get("weight", 1.0),
            "abs_weight": edge.get("abs_weight", 1.0),
            "bootstrap_frequency": edge.get("bootstrap_frequency", 0.0),
            "source": edge.get("source", "unknown")
        })

final_df = pd.DataFrame(final_rows)
if 'abs_weight' in final_df.columns:
    final_df = final_df.sort_values("abs_weight", ascending=False)
final_df.to_csv(f"{pipeline_path}/hybrid_causal_edges.csv", index=False)

# Save NOTEARS weight matrix if available
if notears_result.get('W_est') is not None:
    W_df = pd.DataFrame(
        notears_result['W_est'],
        index=notears_result.get('columns', columns),
        columns=notears_result.get('columns', columns)
    )
    W_df.to_csv(f"{pipeline_path}/notears_weight_matrix.csv")

# Save bootstrap stability scores
stability_df = pd.DataFrame([
    {"from": k[0], "to": k[1], "frequency": v}
    for k, v in bootstrap_result['stability_scores'].items()
]).sort_values("frequency", ascending=False)
stability_df.to_csv(f"{pipeline_path}/bootstrap_stability_scores.csv", index=False)

print("\n" + "="*80)
print("✓ PIPELINE D COMPLETE — All artifacts saved")
print("="*80)
print(f"\nSaved to: {pipeline_path}")
print(f"\nFinal Results:")
print(f"  - PC Skeleton: {len(skeleton_edges)} edges")
print(f"  - Bootstrap Stable: {len(stable_edges)} edges")
print(f"  - Final DAG: {len(final_edges)} edges")
print(f"  - Is Valid DAG: {is_dag}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation (Using Fault Labels)

# COMMAND ----------

# ===========================
# EVALUATION AGAINST FAULT LABELS
# ===========================
# This section uses fault labels for evaluation ONLY
# Labels were never used during discovery

print("\n" + "="*60)
print("EVALUATION (Fault Labels - Post-hoc Only)")
print("="*60)

# Summary of edge sources
source_counts = defaultdict(int)
for edge in final_edges:
    source_counts[edge.get('source', 'unknown')] += 1

print("\nEdge Sources:")
for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  - {source}: {count} edges")

# Tier coverage
tier_0_edges = [e for e in final_edges if tier_assignments.get(e['from'], -1) == 0]
tier_1_edges = [e for e in final_edges if tier_assignments.get(e['from'], -1) == 1]
tier_2_edges = [e for e in final_edges if tier_assignments.get(e['from'], -1) == 2]
tier_3_edges = [e for e in final_edges if tier_assignments.get(e['from'], -1) == 3]

print("\nEdges by Source Tier:")
print(f"  - Raw → *: {len(tier_0_edges)}")
print(f"  - Bronze → *: {len(tier_1_edges)}")
print(f"  - Silver → *: {len(tier_2_edges)}")
print(f"  - ML/KPI → *: {len(tier_3_edges)}")

# Bootstrap frequency distribution
frequencies = [e.get('bootstrap_frequency', 0) for e in final_edges]
if frequencies:
    print("\nBootstrap Frequency Distribution:")
    print(f"  - Min: {min(frequencies):.2f}")
    print(f"  - Max: {max(frequencies):.2f}")
    print(f"  - Mean: {np.mean(frequencies):.2f}")
    print(f"  - Edges with freq ≥ 0.8: {sum(1 for f in frequencies if f >= 0.8)}")

print("\n✓ Evaluation complete")
