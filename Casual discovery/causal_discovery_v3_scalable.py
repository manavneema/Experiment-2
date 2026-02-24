# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline D v3: Scalable Hybrid Causal Discovery
# MAGIC 
# MAGIC This notebook implements a **scalable, academically defensible** causal discovery approach.
# MAGIC 
# MAGIC ## Key Design Principles (NO HARDCODING)
# MAGIC 
# MAGIC 1. **Pattern-Based Structural Priors**: Instead of listing specific edges, we define RULES based on naming patterns
# MAGIC 2. **Correlation-Based Isolation Recovery**: Isolated nodes automatically get edges to their most correlated neighbors
# MAGIC 3. **Softer Tier Constraints**: Only forbid edges that skip 2+ tiers (not all downstream→upstream)
# MAGIC 4. **No Same-Computation Blacklist**: Let the data speak, don't pre-filter based on assumptions
# MAGIC 5. **Lower Discovery Thresholds**: More inclusive edge discovery, then let downstream evaluation filter
# MAGIC 
# MAGIC ## Why This is Academically Defensible
# MAGIC 
# MAGIC | Approach | Problem | Solution |
# MAGIC |----------|---------|----------|
# MAGIC | Hardcoded node whitelist | Not generalizable | Pattern-based rules |
# MAGIC | Manual edge addition | Overfitting to test cases | Automatic correlation recovery |
# MAGIC | Strict tier blacklist | Removes real relationships | Soft constraints (2+ tier jumps only) |
# MAGIC | 60% bootstrap threshold | Too aggressive | 40% threshold |

# COMMAND ----------

# MAGIC %pip install networkx scipy scikit-learn pydot causal-learn

# COMMAND ----------

# Imports
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import time
import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict

from pyspark.sql import functions as F

from scipy.optimize import minimize
from scipy.linalg import expm

from causallearn.search.ConstraintBased.PC import pc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# ===========================
# CONFIGURATION - v3 SCALABLE
# ===========================

PIPELINE_NAME = "Hybrid_PC_NOTEARS_Bootstrap_v3"
METRICS_TABLE = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"

# Data configuration
TOTAL_RUNS = 107
FAULT_CUTOFF_DATE = "2025-12-01"

# ===========================================
# KEY CHANGE 1: HIGHER PC ALPHA (more edges)
# ===========================================
PC_ALPHA_CANDIDATES = [0.10, 0.12, 0.15]  # Was [0.05, 0.07, 0.10]
PC_INDEP_TEST = 'fisherz'

# NOTEARS parameters
NOTEARS_LAMBDA_CANDIDATES = [0.01, 0.02, 0.05]
NOTEARS_MAX_ITER = 100
NOTEARS_H_TOL = 1e-8

# ===========================================
# KEY CHANGE 2: LOWER BOOTSTRAP THRESHOLD
# ===========================================
BOOTSTRAP_RESAMPLES = 100
BOOTSTRAP_EDGE_THRESHOLD = 0.40  # Was 0.60 - now more inclusive

# ===========================================
# KEY CHANGE 3: LESS AGGRESSIVE PREPROCESSING
# ===========================================
CORRELATION_THRESHOLD = 0.95  # Was 0.99 - still remove near-duplicates
VARIANCE_THRESHOLD = 0.0  # Only drop truly constant

# ===========================================
# KEY CHANGE 4: ISOLATION RECOVERY PARAMETERS
# ===========================================
ISOLATION_RECOVERY_ENABLED = True
ISOLATION_MIN_CORRELATION = 0.25  # Min |correlation| to add edge
ISOLATION_MAX_EDGES = 2  # Max edges to add per isolated node

# ===========================================
# KEY CHANGE 5: SOFT TIER CONSTRAINTS
# ===========================================
# Only forbid edges that skip 2+ tiers (Raw→Silver, Bronze→KPI)
# Allow adjacent tier edges in both directions
TIER_JUMP_THRESHOLD = 2  # Was 1 (any downstream→upstream forbidden)

# ===========================================
# KEY CHANGE 6: WEIGHT NORMALIZATION
# ===========================================
# NOTE: Weight normalization should happen in SCORING, not in graph storage.
# The graph should preserve original weights; scoring can normalize as needed.
# Keeping this OFF to preserve raw causal strength information.
NORMALIZE_WEIGHTS_BY_SOURCE = False

# ===========================================
# KEY CHANGE 7: BIDIRECTIONAL EDGE HANDLING
# ===========================================
# When both A→B and B→A appear with high bootstrap frequency, this usually indicates:
# 1. Confounding (hidden common cause)
# 2. Feedback loops
# DISABLED by default: Creates cycles and confuses RCA traversal.
# Enable only if your RCA algorithm handles cycles explicitly.
PRESERVE_BIDIRECTIONAL_EDGES = False
BIDIRECTIONAL_STABILITY_THRESHOLD = 0.50  # Slightly above bootstrap threshold (0.40)

# ===========================================
# KEY CHANGE 8: HUB PENALTY IN GRAPH
# ===========================================
# Store out-degree in edge metadata so scoring can normalize
# This is NOT hardcoding - it's graph metadata
INCLUDE_DEGREE_METADATA = True

# Artifact path
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"
pipeline_path = f"{path}/{PIPELINE_NAME}"

print(f"Pipeline D v3 (Scalable) Configuration:")
print(f"  - PC Alpha Candidates: {PC_ALPHA_CANDIDATES} (HIGHER)")
print(f"  - Bootstrap Threshold: {BOOTSTRAP_EDGE_THRESHOLD} (LOWER)")
print(f"  - Correlation Threshold: {CORRELATION_THRESHOLD}")
print(f"  - Tier Jump Threshold: {TIER_JUMP_THRESHOLD} (SOFTER)")
print(f"  - Isolation Recovery: {ISOLATION_RECOVERY_ENABLED}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pattern-Based Structural Priors (NOT Hardcoded)

# COMMAND ----------

# ===========================
# PATTERN-BASED STRUCTURAL PRIORS
# ===========================
# These are RULES based on naming conventions, NOT specific node lists.
# They apply to ANY metrics that match the pattern.

def generate_pattern_based_priors(columns):
    """
    Generate structural prior edges based on naming PATTERNS, not specific nodes.
    
    This is academically defensible because:
    1. It encodes domain knowledge about data pipeline architecture
    2. It applies uniformly to ALL metrics matching the pattern
    3. It's not tuned to specific test cases
    
    Args:
        columns: List of column names
    
    Returns:
        list: List of (from, to) edge tuples
    """
    priors = []
    column_set = set(columns)
    
    # ===========================================
    # PATTERN 1: raw_X → bronze_X (same metric name across tiers)
    # ===========================================
    # If raw_distance_mean and bronze_distance_km_mean both exist,
    # the raw metric causally precedes the bronze metric.
    
    for col in columns:
        if col.startswith('raw_'):
            suffix = col[4:]  # Remove 'raw_' prefix
            
            # Look for bronze version with similar suffix
            for bronze_col in columns:
                if bronze_col.startswith('bronze_'):
                    bronze_suffix = bronze_col[7:]  # Remove 'bronze_' prefix
                    
                    # Match if suffixes are similar (allow _km suffix differences)
                    if suffix == bronze_suffix or \
                       suffix.replace('_mean', '') in bronze_suffix or \
                       bronze_suffix.replace('_km_mean', '_mean') == suffix:
                        priors.append((col, bronze_col))
    
    # ===========================================
    # PATTERN 2: bronze_X → silver_X (same metric across tiers)
    # ===========================================
    for col in columns:
        if col.startswith('bronze_'):
            suffix = col[7:]
            
            for silver_col in columns:
                if silver_col.startswith('silver_'):
                    silver_suffix = silver_col[7:]
                    
                    if suffix == silver_suffix or \
                       suffix.replace('_mean', '') in silver_suffix:
                        priors.append((col, silver_col))
    
    # ===========================================
    # PATTERN 3: null_count metrics → validation metrics
    # ===========================================
    # raw_null_count_X → bronze validation failure metrics
    
    null_count_cols = [c for c in columns if 'null_count' in c]
    validation_cols = [c for c in columns if any(kw in c for kw in 
                       ['null_primary_key', 'dropped', 'invalid', 'removed'])]
    
    for null_col in null_count_cols:
        # Extract what's being counted as null
        match = re.search(r'null_count_(\w+)', null_col)
        if match:
            null_target = match.group(1)
            
            for val_col in validation_cols:
                # Connect if the validation metric relates to the null target
                if null_target in val_col.lower() or \
                   (null_target == 'unit_id' and 'primary_key' in val_col):
                    priors.append((null_col, val_col))
    
    # ===========================================
    # PATTERN 4: distance/duration → computed metrics
    # ===========================================
    # Metrics like fuel_per_100km, speed, idling_per_100km use distance
    
    distance_cols = [c for c in columns if 'distance' in c and 'mean' in c]
    duration_cols = [c for c in columns if 'duration' in c and 'mean' in c]
    computed_cols = [c for c in columns if any(kw in c for kw in 
                     ['per_100km', 'speed', 'rate'])]
    
    for dist_col in distance_cols:
        for comp_col in computed_cols:
            if 'distance' not in comp_col:  # Don't connect distance to itself
                priors.append((dist_col, comp_col))
    
    for dur_col in duration_cols:
        for comp_col in computed_cols:
            if 'duration' not in comp_col and 'speed' in comp_col:
                priors.append((dur_col, comp_col))
    
    # ===========================================
    # PATTERN 5: ingestion_duration across tiers
    # ===========================================
    ingestion_cols = sorted([c for c in columns if 'ingestion_duration' in c])
    for i in range(len(ingestion_cols) - 1):
        priors.append((ingestion_cols[i], ingestion_cols[i+1]))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_priors = []
    for edge in priors:
        if edge not in seen and edge[0] in column_set and edge[1] in column_set:
            seen.add(edge)
            unique_priors.append(edge)
    
    print(f"Generated {len(unique_priors)} pattern-based structural priors")
    return unique_priors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation-Based Isolation Recovery

# COMMAND ----------

# ===========================
# ISOLATION RECOVERY
# ===========================

# ===========================
# WEIGHT NORMALIZATION
# ===========================

def normalize_edge_weights(edges, method='rank'):
    """
    Normalize edge weights to remove source-based bias.
    
    Problem: Structural priors have weights ~0.3-0.7, bootstrap edges ~0.01-0.02
    This 40x difference causes structural edges to dominate scoring.
    
    Solution: Normalize within source OR use rank-based weights.
    
    Args:
        edges: List of edge dicts
        method: 'rank' (use rank position), 'zscore' (normalize within source), 'minmax' (0-1 scale)
    
    Returns:
        list: Edges with normalized weights
    """
    if not edges:
        return edges
    
    if method == 'rank':
        # Rank-based: higher abs_weight → higher rank → normalized weight
        sorted_edges = sorted(edges, key=lambda x: x.get('abs_weight', 0), reverse=True)
        n = len(sorted_edges)
        for i, edge in enumerate(sorted_edges):
            edge['normalized_weight'] = (n - i) / n  # 1.0 for highest, 1/n for lowest
            edge['weight_rank'] = i + 1
        return sorted_edges
    
    elif method == 'zscore':
        # Z-score within each source type
        by_source = defaultdict(list)
        for edge in edges:
            source = edge.get('source', 'unknown')
            by_source[source].append(edge)
        
        for source, source_edges in by_source.items():
            weights = [e.get('abs_weight', 0) for e in source_edges]
            if len(weights) > 1 and np.std(weights) > 0:
                mean_w = np.mean(weights)
                std_w = np.std(weights)
                for edge in source_edges:
                    edge['normalized_weight'] = (edge.get('abs_weight', 0) - mean_w) / std_w
            else:
                for edge in source_edges:
                    edge['normalized_weight'] = 0.0
        
        return edges
    
    elif method == 'minmax':
        # Min-max to [0, 1] scale
        weights = [e.get('abs_weight', 0) for e in edges]
        min_w, max_w = min(weights), max(weights)
        range_w = max_w - min_w if max_w > min_w else 1.0
        
        for edge in edges:
            edge['normalized_weight'] = (edge.get('abs_weight', 0) - min_w) / range_w
        
        return edges
    
    return edges


# ===========================
# BIDIRECTIONAL EDGE DETECTION
# ===========================

def detect_bidirectional_edges(stability_scores, threshold=0.80):
    """
    Detect edges that appear in both directions with high frequency.
    
    Args:
        stability_scores: Dict mapping (from, to) → frequency
        threshold: Min frequency for both directions to be considered bidirectional
    
    Returns:
        list: List of (nodeA, nodeB, freq_ab, freq_ba) tuples
    """
    bidirectional = []
    seen = set()
    
    for (a, b), freq_ab in stability_scores.items():
        if freq_ab < threshold:
            continue
        
        freq_ba = stability_scores.get((b, a), 0)
        if freq_ba >= threshold:
            pair = tuple(sorted([a, b]))
            if pair not in seen:
                # Return with stability scores for export
                if a < b:
                    bidirectional.append((a, b, freq_ab, freq_ba))
                else:
                    bidirectional.append((b, a, freq_ba, freq_ab))
                seen.add(pair)
    
    return bidirectional


def preserve_bidirectional_edges(edges, bidirectional_pairs, blacklist_set=None):
    """
    For bidirectional pairs, ensure BOTH directions are in the edge list.
    
    IMPORTANT: Respects blacklist - won't add edges that were intentionally removed.
    
    Args:
        edges: Current edge list
        bidirectional_pairs: List of (nodeA, nodeB, freq_ab, freq_ba) tuples
        blacklist_set: Set of (from, to) tuples that are forbidden
    
    Returns:
        list: Updated edges with reverse edges added (if not blacklisted)
    """
    existing = set((e['from'], e['to']) for e in edges)
    added_count = 0
    skipped_blacklist = 0
    
    for item in bidirectional_pairs:
        a, b = item[0], item[1]
        
        has_ab = (a, b) in existing
        has_ba = (b, a) in existing
        
        if has_ab and not has_ba:
            # Check if B→A is blacklisted
            if blacklist_set and (b, a) in blacklist_set:
                skipped_blacklist += 1
                continue
            
            for e in edges:
                if e['from'] == a and e['to'] == b:
                    reverse = {
                        'from': b,
                        'to': a,
                        'weight': e.get('weight', 0),
                        'abs_weight': e.get('abs_weight', 0),
                        'bootstrap_frequency': e.get('bootstrap_frequency', 0),
                        'source': 'bidirectional_reverse',
                        'bidirectional': True
                    }
                    edges.append(reverse)
                    e['bidirectional'] = True
                    added_count += 1
                    break
        
        elif has_ba and not has_ab:
            # Check if A→B is blacklisted
            if blacklist_set and (a, b) in blacklist_set:
                skipped_blacklist += 1
                continue
            
            for e in edges:
                if e['from'] == b and e['to'] == a:
                    reverse = {
                        'from': a,
                        'to': b,
                        'weight': e.get('weight', 0),
                        'abs_weight': e.get('abs_weight', 0),
                        'bootstrap_frequency': e.get('bootstrap_frequency', 0),
                        'source': 'bidirectional_reverse',
                        'bidirectional': True
                    }
                    edges.append(reverse)
                    e['bidirectional'] = True
                    added_count += 1
                    break
    
    print(f"  Added {added_count} reverse edges for bidirectional relationships")
    if skipped_blacklist > 0:
        print(f"  Skipped {skipped_blacklist} (blacklisted)")
    return edges


# ===========================
# HUB DEGREE METADATA
# ===========================

def add_degree_metadata(edges):
    """
    Add in-degree and out-degree to each edge for downstream scoring.
    
    This allows scoring algorithms to normalize by connectivity
    WITHOUT hardcoding specific node penalties.
    
    Args:
        edges: List of edge dicts
    
    Returns:
        list: Edges with degree metadata added
    """
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    
    for edge in edges:
        out_degree[edge['from']] += 1
        in_degree[edge['to']] += 1
    
    for edge in edges:
        edge['source_out_degree'] = out_degree[edge['from']]
        edge['source_in_degree'] = in_degree[edge['from']]
        edge['target_out_degree'] = out_degree[edge['to']]
        edge['target_in_degree'] = in_degree[edge['to']]
    
    return edges


# ===========================
# ISOLATION RECOVERY
# ===========================

def recover_isolated_nodes(edges, data, tier_assignments, 
                           min_correlation=0.25, max_edges_per_node=2):
    """
    Automatically recover isolated nodes by adding edges based on correlation.
    
    This is academically defensible because:
    1. It's a FALLBACK mechanism that only activates for isolated nodes
    2. It uses statistical correlation, not manual specification
    3. It respects tier constraints (only adds edges within same or adjacent tiers)
    4. It's transparent - all recovered edges are marked as 'isolation_recovery'
    
    Args:
        edges: Current list of edges (dicts with 'from' and 'to')
        data: Preprocessed DataFrame
        tier_assignments: Dict mapping column → tier
        min_correlation: Minimum |correlation| to add edge
        max_edges_per_node: Maximum edges to add per isolated node
    
    Returns:
        tuple: (updated edges list, list of recovered edges)
    """
    # Find all connected nodes
    connected = set()
    for edge in edges:
        if isinstance(edge, dict):
            connected.add(edge['from'])
            connected.add(edge['to'])
        else:
            connected.add(edge[0])
            connected.add(edge[1])
    
    # Find isolated nodes
    all_nodes = set(data.columns)
    isolated = all_nodes - connected
    
    if not isolated:
        print("No isolated nodes found - no recovery needed")
        return edges, []
    
    print(f"Found {len(isolated)} isolated nodes - attempting correlation-based recovery")
    
    # Compute correlation matrix
    corr_matrix = data.corr()
    
    recovered_edges = []
    
    for node in sorted(isolated):
        node_tier = tier_assignments.get(node, 2)
        
        # Find top correlated nodes (within tier constraints)
        correlations = corr_matrix[node].drop(node).abs()
        
        # Filter by tier - only allow same tier or adjacent tier
        valid_targets = []
        for target, corr_val in correlations.items():
            target_tier = tier_assignments.get(target, 2)
            tier_diff = abs(node_tier - target_tier)
            
            if tier_diff <= 1 and corr_val >= min_correlation:
                valid_targets.append((target, corr_val, target_tier))
        
        # Sort by correlation and take top max_edges_per_node
        valid_targets.sort(key=lambda x: -x[1])
        top_targets = valid_targets[:max_edges_per_node]
        
        for target, corr_val, target_tier in top_targets:
            # Direction: lower tier → higher tier (or same tier: alphabetical)
            if node_tier < target_tier:
                from_node, to_node = node, target
            elif node_tier > target_tier:
                from_node, to_node = target, node
            else:
                # Same tier - use alphabetical order
                from_node, to_node = (node, target) if node < target else (target, node)
            
            # Estimate weight via OLS
            try:
                X = data[from_node].values.reshape(-1, 1)
                y = data[to_node].values
                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                weight = float(beta[1])
            except:
                weight = float(corr_val)
            
            edge_dict = {
                'from': from_node,
                'to': to_node,
                'weight': weight,
                'abs_weight': abs(weight),
                'bootstrap_frequency': 0.0,  # Not from bootstrap
                'correlation': float(corr_val),
                'source': 'isolation_recovery'  # Clearly marked
            }
            recovered_edges.append(edge_dict)
            edges.append(edge_dict)
            
            print(f"    Recovered: {from_node} → {to_node} (corr={corr_val:.3f})")
    
    print(f"Recovered {len(recovered_edges)} edges for {len(isolated)} isolated nodes")
    return edges, recovered_edges

# COMMAND ----------

# MAGIC %md
# MAGIC ## Soft Tier Constraints

# COMMAND ----------

# ===========================
# SOFT TIER CONSTRAINTS
# ===========================

def generate_soft_tier_blacklist(columns, tier_assignments, tier_jump_threshold=2):
    """
    Generate a SOFT blacklist that only forbids edges skipping 2+ tiers.
    
    This is more principled than forbidding ALL downstream→upstream edges because:
    1. Adjacent tiers CAN have feedback effects (e.g., Bronze→Raw reprocessing)
    2. Only large jumps (Raw→KPI, Bronze→Raw when skipping 2 tiers) are forbidden
    3. Same-tier edges are always allowed
    
    Args:
        columns: List of column names
        tier_assignments: Dict mapping column → tier
        tier_jump_threshold: Only blacklist if |tier_diff| >= this (default: 2)
    
    Returns:
        list: List of (from, to) tuples to blacklist
    """
    blacklist = []
    
    for col_i in columns:
        for col_j in columns:
            if col_i == col_j:
                continue
            
            tier_i = tier_assignments.get(col_i, 2)
            tier_j = tier_assignments.get(col_j, 2)
            
            # Only forbid if jumping 2+ tiers in wrong direction
            # tier_i > tier_j means downstream → upstream
            # We allow adjacent tier edges (tier_diff = 1)
            if tier_i - tier_j >= tier_jump_threshold:
                blacklist.append((col_i, col_j))
    
    print(f"Soft blacklist: {len(blacklist)} edges forbidden (tier jump >= {tier_jump_threshold})")
    return blacklist

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simplified Preprocessing (No Redundancy Detection)

# COMMAND ----------

# ===========================
# SIMPLIFIED PREPROCESSING
# ===========================

def preprocess_simple(df, zscore=True, variance_threshold=0.0, 
                      correlation_threshold=0.95, max_missing_frac=0.3):
    """
    Simple preprocessing WITHOUT aggressive redundancy detection.
    
    Let the causal discovery algorithm decide what's redundant, not heuristics.
    
    Args:
        df: Raw metrics DataFrame
        zscore: Whether to z-score normalize
        variance_threshold: Min variance (0 = only drop constant)
        correlation_threshold: Max correlation for near-duplicates
        max_missing_frac: Max missing fraction before dropping
    
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
        print(f"  Dropped {len(drop_cols)} columns with high missing values")
    
    # Impute remaining
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    imputed_array = imputer.fit_transform(df_num.values)
    df_num = pd.DataFrame(imputed_array, index=df_num.index, columns=df_num.columns)
    
    # Drop ONLY constant columns
    variances = df_num.var()
    const_cols = variances[variances <= variance_threshold].index.tolist()
    meta["dropped_constant"] = const_cols
    
    if const_cols:
        df_num = df_num.drop(columns=const_cols)
        print(f"  Dropped {len(const_cols)} constant columns")
    
    # Remove only near-perfect correlations
    if df_num.shape[1] > 1:
        corr_matrix = df_num.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        removed_corr = []
        remaining = set(df_num.columns)
        
        for col in upper_tri.columns:
            correlated = upper_tri[col][upper_tri[col] > correlation_threshold].index.tolist()
            for corr_col in correlated:
                if corr_col in remaining:
                    removed_corr.append(corr_col)
                    remaining.remove(corr_col)
        
        if removed_corr:
            df_num = df_num.drop(columns=removed_corr)
            print(f"  Dropped {len(removed_corr)} near-duplicate columns (|corr| > {correlation_threshold})")
        
        meta["dropped_high_correlation"] = removed_corr
    
    # Z-score
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
# MAGIC ## Tier Assignment

# COMMAND ----------

def assign_tiers(columns):
    """
    Assign tiers to columns based on naming conventions.
    
    Tier 0: raw_*
    Tier 1: bronze_*
    Tier 2: silver_* (non-ML, non-KPI)
    Tier 3: KPIs and ML outputs
    """
    tier_assignments = {}
    
    for col in columns:
        col_lower = col.lower()
        
        if col.startswith('raw_'):
            tier = 0
        elif col.startswith('bronze_'):
            tier = 1
        elif col.startswith('silver_'):
            if any(kw in col_lower for kw in ['ml_', 'residual', 'prediction', 'imputed']):
                tier = 3
            else:
                tier = 2
        elif any(kw in col_lower for kw in ['fuel_per_100km', 'idling_per_100km', 
                                             'mean_', 'p50_', 'p95_']):
            tier = 3
        else:
            tier = 2
        
        tier_assignments[col] = tier
    
    # Print summary
    tier_counts = defaultdict(int)
    for tier in tier_assignments.values():
        tier_counts[tier] += 1
    
    tier_names = {0: 'Raw', 1: 'Bronze', 2: 'Silver', 3: 'KPI/ML'}
    print("Tier Distribution:")
    for tier in sorted(tier_counts.keys()):
        print(f"  Tier {tier} ({tier_names.get(tier, 'Unknown')}): {tier_counts[tier]}")
    
    return tier_assignments

# COMMAND ----------

# MAGIC %md
# MAGIC ## PC Algorithm

# COMMAND ----------

def extract_pc_skeleton(pc_result, column_names):
    """Extract skeleton edges from PC result."""
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
        return edges
    
    if graph_matrix is None or graph_matrix.shape[0] != len(column_names):
        return edges
    
    seen = set()
    for i in range(len(column_names)):
        for j in range(len(column_names)):
            if i != j and graph_matrix[i, j] != 0:
                edge_value = graph_matrix[i, j]
                
                if edge_value == 1:
                    edges.append((column_names[i], column_names[j], 'directed'))
                elif edge_value == -1:
                    edges.append((column_names[j], column_names[i], 'directed'))
                elif edge_value == 2:
                    pair = tuple(sorted([column_names[i], column_names[j]]))
                    if pair not in seen:
                        edges.append((column_names[i], column_names[j], 'undirected'))
                        seen.add(pair)
    
    return edges


def run_pc_grid_search(df, alpha_candidates, indep_test='fisherz'):
    """Run PC with grid search over alpha."""
    print(f"Running PC Algorithm with α ∈ {alpha_candidates}")
    
    n_samples, n_features = df.shape
    print(f"  Samples: {n_samples}, Features: {n_features}")
    
    results = []
    data = df.values.astype(float)
    
    for alpha in alpha_candidates:
        print(f"\n  Testing α = {alpha}...")
        
        try:
            pc_obj = pc(data, alpha=alpha, indep_test=indep_test)
            edges = extract_pc_skeleton(pc_obj, df.columns.tolist())
            
            n_edges = len(edges)
            avg_edges = 2 * n_edges / n_features if n_features > 0 else 0
            
            print(f"    Edges: {n_edges}, Avg/node: {avg_edges:.2f}")
            
            results.append({
                'alpha': alpha,
                'edges': edges,
                'n_edges': n_edges,
                'avg_edges_per_node': avg_edges,
                'pc_object': pc_obj
            })
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({'alpha': alpha, 'edges': [], 'n_edges': 0, 'error': str(e)})
    
    # Select best (moderate density)
    valid = [r for r in results if r['n_edges'] > 0]
    if not valid:
        return {'method': 'pc-error', 'edges': []}
    
    target_density = 2.5
    for r in valid:
        r['score'] = -abs(r['avg_edges_per_node'] - target_density)
    
    best = max(valid, key=lambda x: x['score'])
    print(f"\n✓ Selected α = {best['alpha']} with {best['n_edges']} edges")
    
    return {
        'method': 'pc-success',
        'alpha': best['alpha'],
        'edges': best['edges'],
        'n_edges': best['n_edges'],
        'avg_edges_per_node': best['avg_edges_per_node'],
        'all_results': [{k: v for k, v in r.items() if k != 'pc_object'} for r in results]
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## NOTEARS

# COMMAND ----------

class NOTEARSConstrained:
    """NOTEARS constrained to skeleton."""
    
    def __init__(self, lambda1=0.01, max_iter=100, h_tol=1e-8):
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.W_est = None
    
    def _loss(self, W, X):
        n, d = X.shape
        R = X - X @ W
        return 0.5 / n * np.trace(R.T @ R)
    
    def _h(self, W):
        d = W.shape[0]
        M = W * W
        E = expm(M)
        return np.trace(E) - d
    
    def _h_grad(self, W):
        M = W * W
        E = expm(M)
        return 2 * W * E
    
    def fit(self, X, skeleton_mask):
        n, d = X.shape
        W = np.random.randn(d, d) * 0.1 * skeleton_mask
        np.fill_diagonal(W, 0)
        
        rho, alpha = 1.0, 0.0
        
        for iteration in range(self.max_iter):
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
            
            result = minimize(objective, W.flatten(), method='L-BFGS-B', jac=gradient,
                            options={'maxiter': 100, 'disp': False})
            
            W = result.x.reshape(d, d) * skeleton_mask
            np.fill_diagonal(W, 0)
            
            h_val = self._h(W)
            if h_val < self.h_tol:
                break
            
            alpha += rho * h_val
            rho = min(rho * 2, 1e16)
        
        self.W_est = W
        self.converged = h_val < self.h_tol
        self.final_h = h_val
        return self


def run_notears_on_skeleton(df, skeleton_edges, lambda_candidates, columns):
    """Run NOTEARS constrained to PC skeleton."""
    print(f"Running NOTEARS with λ ∈ {lambda_candidates}")
    
    n_features = len(columns)
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
    
    print(f"  Skeleton mask has {int(skeleton_mask.sum())} slots")
    
    X = df[columns].values.astype(float)
    results = []
    
    for lambda1 in lambda_candidates:
        print(f"\n  Testing λ = {lambda1}...")
        
        try:
            model = NOTEARSConstrained(lambda1=lambda1, max_iter=NOTEARS_MAX_ITER, h_tol=NOTEARS_H_TOL)
            model.fit(X, skeleton_mask)
            
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
            
            print(f"    Converged: {model.converged}, Edges: {len(weighted_edges)}")
            results.append({
                'lambda1': lambda1,
                'edges': weighted_edges,
                'n_edges': len(weighted_edges),
                'W_est': W,
                'converged': model.converged,
                'final_h': model.final_h
            })
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({'lambda1': lambda1, 'edges': [], 'n_edges': 0, 'error': str(e)})
    
    valid = [r for r in results if r['n_edges'] > 0 and r.get('converged', False)]
    if not valid:
        valid = [r for r in results if r['n_edges'] > 0]
    
    if not valid:
        return {'method': 'notears-error', 'edges': []}
    
    best = min(valid, key=lambda x: abs(x['n_edges'] - len(skeleton_edges) * 0.7))
    print(f"\n✓ Selected λ = {best['lambda1']} with {best['n_edges']} edges")
    
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
# MAGIC ## Bootstrap Stability

# COMMAND ----------

def bootstrap_stability(df, skeleton_edges, notears_lambda, columns, 
                        n_resamples=100, edge_threshold=0.40):
    """Bootstrap edge stability with LOWER threshold."""
    print(f"Bootstrap Stability ({n_resamples} resamples, threshold={edge_threshold})")
    
    n_samples, n_features = df.shape
    col_to_idx = {col: idx for idx, col in enumerate(columns)}
    
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
    
    edge_counts = defaultdict(int)
    edge_weights = defaultdict(list)
    
    X = df[columns].values.astype(float)
    
    for b in range(n_resamples):
        if (b + 1) % 20 == 0:
            print(f"  Bootstrap {b+1}/{n_resamples}...")
        
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        
        try:
            model = NOTEARSConstrained(lambda1=notears_lambda, max_iter=50, h_tol=1e-6)
            model.fit(X_boot, skeleton_mask)
            
            W = model.W_est
            for i in range(n_features):
                for j in range(n_features):
                    if abs(W[i, j]) > 1e-6:
                        edge_key = (columns[i], columns[j])
                        edge_counts[edge_key] += 1
                        edge_weights[edge_key].append(W[i, j])
        except:
            continue
    
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
    
    stable_edges = sorted(stable_edges, key=lambda x: (-x['bootstrap_frequency'], -x['abs_weight']))
    
    print(f"\n✓ Stable edges (freq ≥ {edge_threshold}): {len(stable_edges)}")
    return {
        'stable_edges': stable_edges,
        'stability_scores': stability_scores,
        'n_resamples': n_resamples,
        'threshold': edge_threshold,
        'total_edges_seen': len(edge_counts)
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Graph Utilities

# COMMAND ----------

def apply_blacklist(edges, blacklist_set):
    """Remove blacklisted edges."""
    filtered = []
    removed = []
    
    for edge in edges:
        key = (edge['from'], edge['to']) if isinstance(edge, dict) else (edge[0], edge[1])
        if key in blacklist_set:
            removed.append(edge)
        else:
            filtered.append(edge)
    
    print(f"Blacklist: {len(edges)} → {len(filtered)} edges ({len(removed)} removed)")
    return filtered, removed


def add_pattern_priors(edges, pattern_priors, data, existing_edge_set, blacklist_set=None):
    """Add pattern-based structural priors (respecting blacklist)."""
    print("Adding pattern-based structural priors...")
    
    available_cols = set(data.columns)
    added = []
    skipped_blacklist = 0
    
    for from_col, to_col in pattern_priors:
        if from_col in available_cols and to_col in available_cols:
            if (from_col, to_col) not in existing_edge_set:
                # CHECK BLACKLIST before adding
                if blacklist_set and (from_col, to_col) in blacklist_set:
                    skipped_blacklist += 1
                    continue
                
                try:
                    X = data[from_col].values.reshape(-1, 1)
                    y = data[to_col].values
                    X_int = np.column_stack([np.ones(len(X)), X])
                    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
                    weight = float(beta[1])
                except:
                    weight = 0.0
                
                edge_dict = {
                    'from': from_col,
                    'to': to_col,
                    'weight': weight,
                    'abs_weight': abs(weight),
                    'bootstrap_frequency': 1.0,
                    'source': 'structural_pattern'
                }
                edges.append(edge_dict)
                added.append((from_col, to_col))
    
    print(f"  Added {len(added)} pattern-based priors")
    if skipped_blacklist > 0:
        print(f"  Skipped {skipped_blacklist} (blacklisted)")
    return edges, added


def validate_dag(edges):
    """Check if graph is acyclic."""
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
            print("✓ Valid DAG (no cycles)")
        else:
            print(f"⚠️  Found {len(cycles)} cycles")
        
        return is_dag, cycles
    except Exception as e:
        return True, []


def build_adjacency_maps(edges):
    """Build upstream and downstream maps."""
    upstream = defaultdict(list)
    downstream = defaultdict(list)
    
    for edge in edges:
        if isinstance(edge, dict):
            src, dst = edge['from'], edge['to']
        else:
            src, dst = edge[0], edge[1]
        
        downstream[src].append(dst)
        upstream[dst].append(src)
    
    return dict(upstream), dict(downstream)


def compute_baseline_stats(df):
    """Compute baseline statistics for anomaly detection."""
    stats = {}
    for col in df.columns:
        values = df[col].dropna()
        if len(values) > 0:
            q1, q3 = np.percentile(values, [25, 75])
            stats[col] = {
                'n': len(values),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'median': float(values.median()),
                'q1': float(q1),
                'q3': float(q3),
                'IQR': float(q3 - q1),
                'min': float(values.min()),
                'max': float(values.max())
            }
    return stats

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading

# COMMAND ----------

def load_full_metrics(metrics_sdf, total_runs=107, date_col='date', fault_cutoff='2025-12-01'):
    """Load full metrics matrix."""
    sdf = metrics_sdf.select(
        F.col(date_col).alias('date'),
        F.col('metric_name'),
        F.col('metric_value')
    )
    
    recent_dates = sdf.select('date').distinct().orderBy(F.desc('date')).limit(total_runs)
    recent = sdf.join(recent_dates, on='date', how='inner')
    
    pivot = (recent
             .withColumn('metric_value', F.col('metric_value').cast('double'))
             .groupBy('date')
             .pivot('metric_name')
             .agg(F.first('metric_value')))
    
    pdf = pivot.orderBy('date').toPandas()
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.set_index('date').sort_index()
    
    date_labels = pd.DataFrame({
        'date': pdf.index,
        'is_fault': pdf.index < pd.Timestamp(fault_cutoff)
    }).set_index('date')
    
    n_fault = date_labels['is_fault'].sum()
    n_clean = len(date_labels) - n_fault
    
    print(f"Loaded {len(pdf)} days: {n_fault} fault, {n_clean} clean")
    return pdf, date_labels

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Pipeline

# COMMAND ----------

print("="*80)
print("PIPELINE D v3: SCALABLE HYBRID CAUSAL DISCOVERY")
print("="*80)

# =====================
# PHASE 1: DATA LOADING
# =====================
print("\n" + "="*60)
print("PHASE 1: DATA LOADING")
print("="*60)

metrics_sdf = spark.table(METRICS_TABLE)
metrics_pdf, date_labels = load_full_metrics(
    metrics_sdf, 
    total_runs=TOTAL_RUNS, 
    fault_cutoff=FAULT_CUTOFF_DATE
)

fault_dates = date_labels[date_labels['is_fault']].index.tolist()
clean_dates = date_labels[~date_labels['is_fault']].index.tolist()

# =====================
# PHASE 2: PREPROCESSING (SIMPLIFIED)
# =====================
print("\n" + "="*60)
print("PHASE 2: SIMPLIFIED PREPROCESSING")
print("="*60)

preprocessed_df, preprocess_meta = preprocess_simple(
    metrics_pdf,
    zscore=True,
    variance_threshold=VARIANCE_THRESHOLD,
    correlation_threshold=CORRELATION_THRESHOLD
)

columns = preprocessed_df.columns.tolist()

# =====================
# PHASE 3: TIER ASSIGNMENT & SOFT CONSTRAINTS
# =====================
print("\n" + "="*60)
print("PHASE 3: TIER ASSIGNMENT & SOFT CONSTRAINTS")
print("="*60)

tier_assignments = assign_tiers(columns)
blacklist = generate_soft_tier_blacklist(columns, tier_assignments, TIER_JUMP_THRESHOLD)
blacklist_set = set(blacklist)

# =====================
# PHASE 4: PATTERN-BASED PRIORS
# =====================
print("\n" + "="*60)
print("PHASE 4: PATTERN-BASED STRUCTURAL PRIORS")
print("="*60)

pattern_priors = generate_pattern_based_priors(columns)

# =====================
# PHASE 5: PC SKELETON
# =====================
print("\n" + "="*60)
print("PHASE 5: PC SKELETON DISCOVERY")
print("="*60)

pc_result = run_pc_grid_search(preprocessed_df, PC_ALPHA_CANDIDATES, PC_INDEP_TEST)

if pc_result['method'] == 'pc-error':
    raise Exception("PC Algorithm failed")

skeleton_edges = pc_result['edges']

# =====================
# PHASE 6: NOTEARS WEIGHTS
# =====================
print("\n" + "="*60)
print("PHASE 6: NOTEARS WEIGHT ESTIMATION")
print("="*60)

notears_result = run_notears_on_skeleton(
    preprocessed_df, skeleton_edges, NOTEARS_LAMBDA_CANDIDATES, columns
)

if notears_result['method'] == 'notears-error':
    print("⚠️  NOTEARS failed, using unit weights")
    weighted_edges = [{'from': e[0], 'to': e[1], 'weight': 1.0, 'abs_weight': 1.0}
                      for e in skeleton_edges if len(e) >= 2]
else:
    weighted_edges = notears_result['edges']

# =====================
# PHASE 7: BOOTSTRAP
# =====================
print("\n" + "="*60)
print("PHASE 7: BOOTSTRAP STABILITY")
print("="*60)

bootstrap_result = bootstrap_stability(
    preprocessed_df, skeleton_edges,
    notears_lambda=notears_result.get('lambda1', 0.01),
    columns=columns,
    n_resamples=BOOTSTRAP_RESAMPLES,
    edge_threshold=BOOTSTRAP_EDGE_THRESHOLD
)

stable_edges = bootstrap_result['stable_edges']

# =====================
# PHASE 8: REFINEMENT
# =====================
print("\n" + "="*60)
print("PHASE 8: GRAPH REFINEMENT")
print("="*60)

# Apply soft blacklist
print("\n[Step 8.1] Applying soft tier constraints...")
filtered_edges, removed = apply_blacklist(stable_edges, blacklist_set)

# Add pattern-based priors
print("\n[Step 8.2] Adding pattern-based priors...")
existing_set = set((e['from'], e['to']) for e in filtered_edges)
final_edges, added_patterns = add_pattern_priors(
    filtered_edges, pattern_priors, preprocessed_df, existing_set, blacklist_set
)

# Validate DAG
print("\n[Step 8.3] Validating DAG...")
is_dag, cycles = validate_dag(final_edges)

if not is_dag:
    print("Removing edges to break cycles...")
    for cycle in cycles:
        cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        min_edge = None
        min_weight = float('inf')
        for e in final_edges:
            if (e['from'], e['to']) in cycle_edges and e['abs_weight'] < min_weight:
                min_weight = e['abs_weight']
                min_edge = (e['from'], e['to'])
        if min_edge:
            final_edges = [e for e in final_edges if (e['from'], e['to']) != min_edge]
    is_dag, _ = validate_dag(final_edges)

# =====================
# PHASE 9: ISOLATION RECOVERY
# =====================
print("\n" + "="*60)
print("PHASE 9: ISOLATION RECOVERY")
print("="*60)

if ISOLATION_RECOVERY_ENABLED:
    final_edges, recovered_edges = recover_isolated_nodes(
        final_edges, preprocessed_df, tier_assignments,
        min_correlation=ISOLATION_MIN_CORRELATION,
        max_edges_per_node=ISOLATION_MAX_EDGES
    )
else:
    recovered_edges = []
    print("Isolation recovery disabled")

# =====================
# PHASE 10: BIDIRECTIONAL EDGE PRESERVATION
# =====================
print("\n" + "="*60)
print("PHASE 10: BIDIRECTIONAL EDGE PRESERVATION")
print("="*60)

bidirectional_pairs = []
if PRESERVE_BIDIRECTIONAL_EDGES:
    stability_scores = bootstrap_result.get('stability_scores', {})
    
    # Detect bidirectional relationships
    bidirectional_pairs = detect_bidirectional_edges(
        stability_scores, 
        threshold=BIDIRECTIONAL_STABILITY_THRESHOLD
    )
    
    if bidirectional_pairs:
        final_edges = preserve_bidirectional_edges(final_edges, bidirectional_pairs, blacklist_set)
        # Note: Bidirectional edges create cycles - graph is no longer a DAG
        # This is intentional for richer traversal in RCA
    
    print(f"  Detected {len(bidirectional_pairs)} bidirectional pairs")
    print(f"  Edges after preservation: {len(final_edges)}")
else:
    print("Bidirectional preservation disabled")

# Re-check DAG status after all modifications
is_dag_final, cycles_final = validate_dag(final_edges)
if not is_dag_final and bidirectional_pairs:
    print(f"  Note: Graph has cycles due to bidirectional edges (expected)")
is_dag = is_dag_final  # Update for export

# =====================
# PHASE 11: WEIGHT NORMALIZATION
# =====================
print("\n" + "="*60)
print("PHASE 11: WEIGHT NORMALIZATION")
print("="*60)

if NORMALIZE_WEIGHTS_BY_SOURCE:
    # Normalize weights to remove structural prior bias
    final_edges = normalize_edge_weights(final_edges, method='rank')
    print(f"  Normalized {len(final_edges)} edge weights using rank method")
    
    # Verify normalization
    weights = [e.get('normalized_weight', e.get('abs_weight', 0)) for e in final_edges]
    if weights:
        print(f"  Weight range after normalization: [{min(weights):.3f}, {max(weights):.3f}]")
else:
    print("Weight normalization disabled")

# =====================
# PHASE 12: DEGREE METADATA
# =====================
print("\n" + "="*60)
print("PHASE 12: HUB DEGREE METADATA")
print("="*60)

# Always compute degrees (needed for hub_analysis export)
out_degrees = defaultdict(int)
in_degrees = defaultdict(int)
for e in final_edges:
    out_degrees[e['from']] += 1
    in_degrees[e['to']] += 1

if INCLUDE_DEGREE_METADATA:
    final_edges = add_degree_metadata(final_edges)
    
    high_out_hubs = [n for n, d in out_degrees.items() if d >= 3]
    sink_nodes = [n for n, d in in_degrees.items() if d >= 2 and out_degrees.get(n, 0) == 0]
    
    print(f"  High out-degree hubs (≥3): {len(high_out_hubs)}")
    for h in sorted(high_out_hubs, key=lambda x: -out_degrees[x])[:5]:
        print(f"    - {h}: out={out_degrees[h]}, in={in_degrees.get(h, 0)}")
    
    print(f"  Sink nodes (in≥2, out=0): {len(sink_nodes)}")
    for s in sorted(sink_nodes, key=lambda x: -in_degrees[x])[:5]:
        print(f"    - {s}: in={in_degrees[s]}")
else:
    print("Degree metadata disabled")

# =====================
# SUMMARY
# =====================
print("\n" + "="*80)
print("PIPELINE D v3: SUMMARY")
print("="*80)

print(f"\nResults:")
print(f"  - PC Skeleton edges: {len(skeleton_edges)}")
print(f"  - Bootstrap stable edges: {len(stable_edges)}")
print(f"  - Removed by blacklist: {len(removed)}")
print(f"  - Pattern-based priors added: {len(added_patterns)}")
print(f"  - Isolated nodes recovered: {len(recovered_edges)}")
print(f"  - FINAL edges: {len(final_edges)}")
print(f"  - Valid DAG: {is_dag}")

# =====================
# COVERAGE ANALYSIS
# =====================
upstream_map, downstream_map = build_adjacency_maps(final_edges)
connected = set(upstream_map.keys()) | set(downstream_map.keys())
for targets in downstream_map.values():
    connected.update(targets)

isolated_final = set(columns) - connected
print(f"\nCoverage:")
print(f"  - Connected nodes: {len(connected)}/{len(columns)} ({len(connected)/len(columns)*100:.1f}%)")
print(f"  - Still isolated: {len(isolated_final)}")
if isolated_final:
    print(f"    {sorted(isolated_final)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Artifacts

# COMMAND ----------

print("\n" + "="*60)
print("EXPORTING ARTIFACTS")
print("="*60)

baseline_stats = compute_baseline_stats(preprocessed_df)
upstream_map, downstream_map = build_adjacency_maps(final_edges)

dbutils.fs.mkdirs(pipeline_path)

# Core artifacts
artifacts = {
    "pipeline": PIPELINE_NAME,
    "method": "hybrid_pc_notears_bootstrap_v3_scalable",
    "version": "v3",
    "status": "SUCCESS",
    "created_at": datetime.utcnow().isoformat(),
    "config": {
        "pc_alpha_candidates": PC_ALPHA_CANDIDATES,
        "bootstrap_threshold": BOOTSTRAP_EDGE_THRESHOLD,
        "tier_jump_threshold": TIER_JUMP_THRESHOLD,
        "isolation_recovery": ISOLATION_RECOVERY_ENABLED,
        "isolation_min_correlation": ISOLATION_MIN_CORRELATION,
        "normalize_weights": NORMALIZE_WEIGHTS_BY_SOURCE,
        "preserve_bidirectional": PRESERVE_BIDIRECTIONAL_EDGES,
        "bidirectional_threshold": BIDIRECTIONAL_STABILITY_THRESHOLD,
        "include_degree_metadata": INCLUDE_DEGREE_METADATA
    },
    "data_info": {
        "n_fault_dates": len(fault_dates),
        "n_clean_dates": len(clean_dates),
        "fault_dates": [str(d.date()) for d in fault_dates],
        "clean_dates": [str(d.date()) for d in clean_dates]
    },
    "preprocess_meta": preprocess_meta,
    "pc_result": {
        "method": pc_result["method"],
        "alpha_selected": pc_result.get("alpha"),
        "n_skeleton_edges": len(skeleton_edges)
    },
    "notears_result": {
        "method": notears_result.get("method"),
        "lambda_selected": notears_result.get("lambda1"),
        "n_weighted_edges": len(weighted_edges)
    },
    "bootstrap_result": {
        "n_stable_edges": len(stable_edges),
        "threshold": BOOTSTRAP_EDGE_THRESHOLD
    },
    "refinement": {
        "n_removed_by_blacklist": len(removed),
        "n_pattern_priors_added": len(added_patterns),
        "n_isolated_recovered": len(recovered_edges),
        "n_bidirectional_pairs": len(bidirectional_pairs)
    },
    "final_graph": {
        "n_edges": len(final_edges),
        "is_dag": is_dag,
        "n_connected_nodes": len(connected),
        "n_isolated_nodes": len(isolated_final)
    },
    "tier_assignments": tier_assignments
}

# Save artifacts
preprocessed_df.to_csv(f"{pipeline_path}/causal_metrics_matrix.csv")
dbutils.fs.put(f"{pipeline_path}/causal_artifacts.json", json.dumps(artifacts, indent=2, default=str), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/baseline_stats.json", json.dumps(baseline_stats, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/upstream_map.json", json.dumps(upstream_map, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/downstream_map.json", json.dumps(downstream_map, indent=2), overwrite=True)
dbutils.fs.put(f"{pipeline_path}/tier_assignments.json", json.dumps(tier_assignments, indent=2), overwrite=True)

# Edge CSVs
pd.DataFrame([{"from": e[0], "to": e[1], "type": e[2] if len(e) > 2 else "undirected"} 
              for e in skeleton_edges]).to_csv(f"{pipeline_path}/pc_skeleton_edges.csv", index=False)

pd.DataFrame(stable_edges).to_csv(f"{pipeline_path}/bootstrap_stable_edges.csv", index=False)

final_df = pd.DataFrame(final_edges)
if 'abs_weight' in final_df.columns:
    final_df = final_df.sort_values("abs_weight", ascending=False)
final_df.to_csv(f"{pipeline_path}/hybrid_causal_edges.csv", index=False)

if recovered_edges:
    pd.DataFrame(recovered_edges).to_csv(f"{pipeline_path}/recovered_edges.csv", index=False)

if added_patterns:
    pd.DataFrame([{"from": a, "to": b} for a, b in added_patterns]).to_csv(
        f"{pipeline_path}/pattern_prior_edges.csv", index=False)

# Save bidirectional pairs
if bidirectional_pairs:
    pd.DataFrame([{"node_a": a, "node_b": b, "stability_ab": sab, "stability_ba": sba} 
                  for a, b, sab, sba in bidirectional_pairs]).to_csv(
        f"{pipeline_path}/bidirectional_pairs.csv", index=False)

# Save hub analysis
hub_analysis = {
    "high_out_degree_hubs": [{"node": n, "out_degree": out_degrees.get(n, 0), "in_degree": in_degrees.get(n, 0)} 
                            for n in sorted(out_degrees, key=lambda x: -out_degrees[x])[:10]],
    "sink_nodes": [{"node": n, "in_degree": in_degrees.get(n, 0)} 
                   for n in sorted(in_degrees, key=lambda x: -in_degrees[x]) 
                   if out_degrees.get(n, 0) == 0][:10]
} if INCLUDE_DEGREE_METADATA else {}

if hub_analysis:
    dbutils.fs.put(f"{pipeline_path}/hub_analysis.json", json.dumps(hub_analysis, indent=2), overwrite=True)

print(f"\n✓ Artifacts saved to: {pipeline_path}")
print(f"\nFinal Graph Stats:")
print(f"  - Edges: {len(final_edges)}")
print(f"  - Coverage: {len(connected)/len(columns)*100:.1f}%")
print(f"  - Isolated: {len(isolated_final)}")
