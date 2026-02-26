# Databricks notebook source
# MAGIC %pip install networkx scipy scikit-learn pydot

# COMMAND ----------

pip install statsmodels

# COMMAND ----------

pip install causal-learn

# COMMAND ----------

# MAGIC %pip install causal-discovery

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

# statsmodels availability check
try:
    from causal_discovery.algos.notears import NoTears
    has_notears = True
except ImportError:
    has_notears = False

# statsmodels availability check
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    has_statsmodels = True
except ImportError:
    has_statsmodels = False

# ML Libs
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLasso

# Visualisation
import networkx as nx
import matplotlib.pyplot as plt


# COMMAND ----------

def generate_stage_blacklist(metric_cols):
    """Generate conservative blacklist pairs from metric column names.

    Rules encoded:
    - forbid any edge from `silver_` metrics to `bronze_` or `raw_` metrics
    - forbid any edge from `bronze_` metrics to `raw_` metrics

    metric_cols: iterable of metric column names (strings)
    returns: list of (from_metric, to_metric) tuples
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
    feature_sample_ratio: float = 3.0,   # n_feats > n_samples * ratio → reduce
    min_keep_features: int = 50,
):
    """
    Preprocess metrics matrix for causal discovery.

    Steps:
    - normalize null tokens
    - coerce to numeric
    - drop all-null & high-missing columns
    - drop constant columns
    - add missingness indicators
    - impute numeric values
    - optional z-scoring
    - optional feature reduction when features >> samples

    Returns
    -------
    df_clean : pd.DataFrame
    meta : dict
    """
    meta = {}

    # --------------------------------------------------
    # 0) Normalize obvious null tokens
    # --------------------------------------------------
    df = df.replace(["null", "NULL", "None", ""], np.nan)

    # --------------------------------------------------
    # 1) Coerce to numeric (fast path)
    # --------------------------------------------------
    df_num = df.copy()
    for c in df_num.columns:
        if not pd.api.types.is_numeric_dtype(df_num[c]):
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    meta["initial_shape"] = df_num.shape

    # --------------------------------------------------
    # 2) Missingness stats (single pass)
    # --------------------------------------------------
    na_mask = df_num.isna()
    miss_frac = na_mask.mean()

    meta["missing_fraction"] = miss_frac.to_dict()

    # --------------------------------------------------
    # 3) Drop fully-null & high-missing columns
    # --------------------------------------------------
    drop_all_null = miss_frac[miss_frac == 1.0].index.tolist()
    drop_high_missing = miss_frac[miss_frac > max_missing_frac].index.tolist()

    drop_cols = sorted(set(drop_all_null + drop_high_missing))
    meta["dropped_all_null"] = drop_all_null
    meta["dropped_high_missing"] = drop_high_missing

    if drop_cols:
        df_num = df_num.drop(columns=drop_cols)
        na_mask = na_mask.drop(columns=drop_cols)

    # --------------------------------------------------
    # 4) Drop constant columns
    # --------------------------------------------------
    nunique = df_num.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    meta["dropped_constant"] = const_cols

    if const_cols:
        df_num = df_num.drop(columns=const_cols)
        na_mask = na_mask.drop(columns=const_cols)

    if df_num.shape[1] == 0:
        meta["final_shape"] = (df_num.shape[0], 0)
        return pd.DataFrame(index=df.index), meta

    # --------------------------------------------------
    # 5) Missingness indicators (partial missing only)
    # --------------------------------------------------
    partial_missing_cols = miss_frac[
        (miss_frac > 0.0) & (miss_frac <= max_missing_frac)
    ].index.intersection(df_num.columns)

    if len(partial_missing_cols) > 0:
        indicators = na_mask[partial_missing_cols].astype("float32")
        indicators.columns = [f"missing_{c}" for c in indicators.columns]
    else:
        indicators = None

    meta["num_missing_indicators"] = 0 if indicators is None else indicators.shape[1]

    # --------------------------------------------------
    # 6) Impute numeric values
    # --------------------------------------------------
    imputer = SimpleImputer(strategy=impute_strategy)
    imputed_array = imputer.fit_transform(df_num.values)

    # --------------------------------------------------
    # 7) Scale (z-score)
    # --------------------------------------------------
    if zscore:
        scaler = StandardScaler()
        imputed_array = scaler.fit_transform(imputed_array)

    df_clean = pd.DataFrame(
        imputed_array,
        index=df_num.index,
        columns=df_num.columns,
    )

    # --------------------------------------------------
    # 8) Feature reduction if features >> samples
    # --------------------------------------------------
    n_samples, n_feats = df_clean.shape
    meta["pre_reduction_shape"] = (n_samples, n_feats)

    if n_feats > n_samples * feature_sample_ratio:
        var = df_clean.var().sort_values(ascending=False)
        keep_k = max(int(n_samples * 2), min_keep_features)
        keep_cols = var.index[:keep_k]

        df_clean = df_clean[keep_cols]
        meta["feature_reduction"] = {
            "applied": True,
            "kept_features": len(keep_cols),
            "dropped_features": n_feats - len(keep_cols),
        }
    else:
        meta["feature_reduction"] = {"applied": False}

    # --------------------------------------------------
    # 9) Reattach indicators (unscaled)
    # --------------------------------------------------
    if indicators is not None:
        df_clean = pd.concat([df_clean, indicators], axis=1)

    # --------------------------------------------------
    # 10) Final checks
    # --------------------------------------------------
    assert not df_clean.isna().any().any(), "NaNs remain after preprocessing"

    meta["final_shape"] = df_clean.shape

    return df_clean, meta


# COMMAND ----------

def add_temporal_features_for_causality(
    df: pd.DataFrame,
    *,
    lags: int = 1,
    rolling_windows: list[int] | None = None,
    drop_invalid_rows: bool = True,
    prune_high_corr: bool = False,
    corr_threshold: float = 0.99,
):
    """
    Add lagged and rolling temporal features in a PC-safe way.

    Steps:
    - add lag-k features
    - add rolling mean features
    - drop rows invalidated by lagging
    - optionally prune highly collinear temporal features

    Returns
    -------
    df_out : pd.DataFrame
        Temporal feature matrix (no NaNs if drop_invalid_rows=True)
    meta : dict
        Metadata about temporal features added/dropped
    """
    meta = {
        "lags": lags,
        "rolling_windows": rolling_windows or [],
        "dropped_rows": 0,
        "dropped_high_corr_cols": [],
    }

    out = df.copy()

    # -------------------------
    # 1) Lag features
    # -------------------------
    for lag in range(1, lags + 1):
        shifted = df.shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in df.columns]
        out = pd.concat([out, shifted], axis=1)

    # -------------------------
    # 2) Rolling mean features
    # -------------------------
    if rolling_windows:
        for w in rolling_windows:
            rolled = df.rolling(window=w, min_periods=w).mean()
            rolled.columns = [f"{c}_rw{w}" for c in df.columns]
            out = pd.concat([out, rolled], axis=1)

    # -------------------------
    # 3) Drop rows invalidated by lagging
    # -------------------------
    if drop_invalid_rows:
        before = out.shape[0]
        out = out.dropna(axis=0)
        meta["dropped_rows"] = before - out.shape[0]

    # -------------------------
    # 4) Optional: prune highly collinear temporal features
    # -------------------------
    if prune_high_corr and out.shape[1] > 1:
        corr = out.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [
            col for col in upper.columns
            if any(upper[col] > corr_threshold)
        ]

        if to_drop:
            out = out.drop(columns=to_drop)
            meta["dropped_high_corr_cols"] = to_drop

    return out, meta


# COMMAND ----------

# Run PC using causal-learn
def run_constraint_discovery(df: pd.DataFrame, alpha=0.05):
    """Run PC using causal-learn and return the PC object.

    Note: `pc` output is returned in `pc_object`. Converting to a list of
    edges depends on the downstream representation the user prefers; the
    returned `pc_object` can be inspected or stringified in artifacts.
    """
    print('Running PC from causal-learn (ensure variables are numeric and non-constant).')
    try:
        data = df.values
        pc_obj = pc(data, alpha=alpha)
        return {'method': 'causal-learn-pc', 'pc_object': pc_obj, 'edges': None}
    except Exception as e:
        return {'method': 'causal-learn-pc-error', 'error': str(e), 'edges': None}


# COMMAND ----------

def run_precision_skeleton(
    df: pd.DataFrame,
    *,
    alpha: float = 0.01,
    pcor_threshold: float = 0.1,
):
    """
    Fallback skeleton discovery using precision matrix (partial correlations).

    Parameters
    ----------
    df : pd.DataFrame
        Input matrix (rows=samples, cols=features), no NaNs.
    alpha : float
        Regularization strength for GraphicalLasso.
    pcor_threshold : float
        Absolute partial correlation threshold for edge inclusion.

    Returns
    -------
    dict with keys:
        - method
        - edges
        - partial_correlations
    """
    X = df.values
    cols = df.columns.tolist()

    try:
        model = GraphicalLasso(alpha=alpha, max_iter=1000)
        model.fit(X)

        precision = model.precision_

        # Convert precision to partial correlations
        d = np.sqrt(np.diag(precision))
        pcor = -precision / np.outer(d, d)
        np.fill_diagonal(pcor, 0.0)

        edges = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = pcor[i, j]
                if abs(val) >= pcor_threshold:
                    edges.append({
                        "from": cols[i],
                        "to": cols[j],
                        "partial_corr": float(val),
                    })

        return {
            "method": "precision-partial-corr",
            "edges": edges,
            "partial_correlations": pcor,
        }

    except Exception as e:
        return {
            "method": "precision-skeleton-error",
            "error": str(e),
            "edges": None,
        }


# COMMAND ----------

def run_notears(df: pd.DataFrame, l1=0.1, max_iter=100):
    """
    Run NOTEARS for global orientation.
    Output is interpreted as directional preference, not ground truth.
    """
    if not has_notears:
        return {"method": "notears-unavailable", "adj": None}

    X = df.values.astype(float)

    # The NoTears API differs across packages. Try several common constructor
    # and fit signatures and extract an adjacency matrix robustly.
    try:
        model = None

        # Preferred: no-arg constructor, fit with lambda1 and max_iter
        try:
            model = NoTears()
            try:
                model.fit(X, lambda1=l1, max_iter=max_iter)
            except TypeError:
                # some implementations don't accept max_iter in fit
                try:
                    model.fit(X, lambda1=l1)
                except TypeError:
                    model.fit(X)

        except TypeError:
            # Fallback: constructor accepts kwargs (but may not in your version)
            try:
                model = NoTears(max_iter=max_iter)
                try:
                    model.fit(X, lambda1=l1)
                except TypeError:
                    model.fit(X)
            except Exception:
                # Last resort: try calling a functional API if exposed
                raise

        # Try common attribute names for adjacency
        W_est = None
        for attr in ("W_est", "W", "adj", "W_hat"):
            if hasattr(model, attr):
                W_est = getattr(model, attr)
                break

        # Some implementations return numpy arrays from fit or expose a method
        if W_est is None:
            # try .get_adj() or model.adj_matrix
            for fn in ("get_adj", "adj_matrix"):
                if hasattr(model, fn):
                    try:
                        W_est = getattr(model, fn)()
                        break
                    except Exception:
                        pass

        if W_est is None:
            return {"method": "notears-error", "error": "No adjacency matrix found after NoTears fit", "adj": None}

        return {"method": "notears", "adj": W_est, "columns": df.columns.tolist()}

    except Exception as e:
        return {"method": "notears-error", "error": str(e), "adj": None}

# COMMAND ----------

def granger_check(
    time_df: pd.DataFrame,
    candidate_pairs,
    maxlag=1,
    alpha=0.05,
):
    results = {}

    if not has_statsmodels:
        print("statsmodels not available; skipping Granger checks")
        return results

    cols = set(time_df.columns)

    for a, b, _ in candidate_pairs:
        out = {}

        # Skip if columns not present
        if a not in cols or b not in cols:
            out["skipped"] = "columns_not_present"
            results[(a, b)] = out
            continue

        try:
            # a -> b
            df_ab = time_df[[b, a]].dropna()
            if len(df_ab) >= maxlag + 5:
                res_ab = grangercausalitytests(
                    df_ab, maxlag=maxlag, verbose=False
                )
                pvals_ab = [
                    res_ab[i + 1][0]["ssr_ftest"][1] for i in range(maxlag)
                ]
                out["pvals_a_to_b"] = pvals_ab
                out["a_causes_b"] = np.min(pvals_ab) < alpha
            else:
                out["a_causes_b"] = None

            # b -> a
            df_ba = time_df[[a, b]].dropna()
            if len(df_ba) >= maxlag + 5:
                res_ba = grangercausalitytests(
                    df_ba, maxlag=maxlag, verbose=False
                )
                pvals_ba = [
                    res_ba[i + 1][0]["ssr_ftest"][1] for i in range(maxlag)
                ]
                out["pvals_b_to_a"] = pvals_ba
                out["b_causes_a"] = np.min(pvals_ba) < alpha
            else:
                out["b_causes_a"] = None

        except Exception as e:
            out["error"] = str(e)

        results[(a, b)] = out

    return results

# COMMAND ----------

# Human priors application and visualization helpers
def apply_human_priors(skeleton_edges, blacklist, whitelist):
    """Apply simple human priors to the skeleton edges.

    - `skeleton_edges` expected as list of tuples (a,b,score)
    - `blacklist` is list of (from, to) tuples to remove
    - `whitelist` is list of (from, to) tuples to force present

    Returns filtered_edges, and a dict with actions taken for review.
    """
    kept = []
    removed = []
    present = set((a,b) for a,b,_ in skeleton_edges)

    # Remove blacklisted edges
    for a,b,score in skeleton_edges:
        if (a,b) in blacklist or (a,b) in [(x,y) for x,y in blacklist]:
            removed.append((a,b,score,'blacklisted'))
        else:
            kept.append((a,b,score))

    # Add whitelisted edges if missing (score=None)
    added = []
    for a,b in whitelist:
        if (a,b) not in present:
            added.append((a,b,None,'whitelisted'))
            kept.append((a,b,None))

    review = {'removed': removed, 'added': added}
    return kept, review


import matplotlib.pyplot as plt
import networkx as nx

def visualize_skeleton(edges, top_k=100):
    """
    Display a simple networkx plot of skeleton edges inline.
    Edges is list of (a, b, score).
    """
    if not edges:
        print("No edges to visualize")
        return None

    # Prefer edges with scores
    scored = [e for e in edges if e[2] is not None]
    scored.sort(key=lambda x: -abs(x[2]))
    show_edges = scored[:top_k] if scored else edges[:top_k]

    G = nx.DiGraph()
    for a, b, score in show_edges:
        w = 1.0 if score is None else float(score)
        G.add_edge(a, b, weight=w)

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42)

    weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
    widths = [max(0.5, w * 2) for w in weights]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        font_size=8,
        width=widths,
        alpha=0.85,
    )

    plt.title("Candidate Causal Skeleton (Top Edges)")
    plt.tight_layout()
    plt.show()

    return G

# COMMAND ----------

def spark_metrics_to_matrix(spark, metrics_table, max_runs=180, date_col='date'):
    """Read metrics table and pivot to wide pandas DataFrame (rows=dates, cols=metric_name)."""
    sdf = spark.table(metrics_table).select(F.col(date_col).alias('date'), F.col('metric_name'), F.col('metric_value'))
    # keep max_runs most recent dates
    recent_dates = sdf.select('date').distinct().orderBy(F.desc('date')).limit(max_runs)
    recent = sdf.join(recent_dates, on='date', how='inner')
    # pivot and aggregate (cast values to double)
    pivot = (recent
             .withColumn('metric_value', F.col('metric_value').cast('double'))
             .groupBy('date')
             .pivot('metric_name')
             .agg(F.first('metric_value')))
    pdf = pivot.orderBy('date').toPandas()
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.set_index('date').sort_index()
    return pdf

# COMMAND ----------

# Defining Configuration
METRICS_TABLE = "bms_ds_prod.bms_ds_dasc.temp_raw_metrics"

# Parameters
MIN_RUNS_FOR_NOTEARS = 30
CORR_THRESHOLD = 0.3

# Safety guard: max runs to pivot in Spark to avoid heavy full-history scans
MAX_RUNS_TO_PIVOT = 180

# DBFS path
path = "/Volumes/bms_ds_science_prod/bms_ds_dasc/bms_ds_dasc/lab_day/causal_discovery_artifacts"

HUMAN_PRIOR_WHITELIST = [
    (
        "raw_input_record_count",
        "bronze_input_rows",
    ),  # raw record count is the direct input to cleaning -> bronze rows
    (
        "raw_ingestion_duration_sec",
        "bronze_ingestion_duration_sec",
    ),  # ingestion time propagates downstream
    (
        "raw_min_trip_start_ts",
        "bronze_input_rows",
    ),  # run temporal window limits which raw rows are included
    (
        "raw_max_trip_end_ts",
        "bronze_output_rows",
    ),  # run temporal end affects which trips survive cleaning
    (
        "raw_distance_mean",
        "bronze_distance_km_mean",
    ),  # same underlying distance aggregate (unit/derivation change)
    (
        "raw_avg_speed_mean",
        "silver_avg_speed_imputed",
    ),  # avg-speed upstream used for downstream imputation/derived speed metrics
    (
        "raw_unique_units",
        "bronze_output_rows",
    ),  # unique unit count influences dedup and output volume
    (
        "raw_null_count_unit_id",
        "bronze_null_primary_key_rows",
    ),  # null unit_id upstream -> primary-key nulls in bronze
    (
        "bronze_input_rows",
        "silver_input_data_count",
    ),  # bronze input volume becomes silver's input count
    (
        "bronze_output_rows",
        "silver_count_after_feature_engineering",
    ),  # bronze output flows into feature-engineering counts
    (
        "bronze_survival_rate",
        "silver_survival_rate",
    ),  # survival fraction defined upstream drives downstream survival stat
    (
        "bronze_distance_km_mean",
        "silver_avg_speed_imputed",
    ),  # distance/duration aggregates inform speed imputation downstream
    (
        "bronze_duration_mean",
        "silver_avg_speed_imputed",
    ),  # duration upstream used for avg-speed imputation downstream
    (
        "bronze_ingestion_duration_sec",
        "silver_ingestion_duration_sec",
    ),  # ingestion runtime carries forward
    (
        "bronze_distance_km_mean",
        "mean_fuel_per_100km",
    ),  # fuel-per-100km KPIs depend on distance aggregates
    (
        "bronze_distance_km_mean",
        "p50_fuel_per_100km",
    ),  # percentile KPIs depend on distance/fuel aggregates
    ("bronze_distance_km_mean", "p95_fuel_per_100km"),  #
    (
        "silver_ml_imputed_fuel_mean",
        "silver_ml_prediction_mean",
    ),  # imputed-fuel mean and model prediction mean are closely related (confidence: medium-high)
    (
        "silver_ml_residual_mean",
        "silver_ml_abs_residual_mean",
    ),  # residual mean and absolute-residual mean are derived from same residuals
]

metric_cols = (
    spark.table("bms_ds_prod.bms_ds_dasc.temp_raw_metrics")
    .select("metric_name")
    .distinct()
    .rdd.flatMap(lambda x: x)
    .collect()
)
HUMAN_PRIOR_BLACKLIST = generate_stage_blacklist(metric_cols)

# COMMAND ----------

# def run_baseline_pipeline(spark):
print("Loading metrics (step 1)")
metrics_pdf = spark_metrics_to_matrix(
    spark, metrics_table=METRICS_TABLE
)

print("Preprocessing metrics matrix (step 2)")
scaled, preprocess_meta = preprocess_metrics_matrix(metrics_pdf)

print("Adding temporal features (step 3)")
lagged, lag_meta = add_temporal_features_for_causality(
    scaled,
    lags=1,
    rolling_windows=[3],
    prune_high_corr=False,   # keep off for first runs
)

# -------------------------
# Step 3.5: variance filter
# -------------------------
print("Filtering low-variance columns")
var = lagged.var()
lagged = lagged[var[var > 1e-6].index]

# -------------------------
# Step 3.6: feature cap (PC-safe)
# -------------------------
n_samples, n_feats = lagged.shape
max_feats = max(int(n_samples / 2), 10)

if n_feats > max_feats:
    var = lagged.var().sort_values(ascending=False)
    lagged = lagged[var.index[:max_feats]]
    print(f"Reduced features to {lagged.shape[1]} for PC stability")

# -------------------------
# Step 4: skeleton discovery
# -------------------------
print("Computing causal skeleton (step 4)")
skeleton_res = run_constraint_discovery(lagged)

if skeleton_res.get("edges") is None:
    print("PC failed — falling back to precision-matrix skeleton")
    skeleton_res = run_precision_skeleton(lagged)

print("Skeleton method:", skeleton_res.get("method"))
print("Num edges:", 0 if skeleton_res.get("edges") is None else len(skeleton_res["edges"]))

# -------------------------
# Step 4.5: extract edges
# -------------------------
skeleton_edges = []

if skeleton_res.get("method") == "precision-partial-corr":
    for e in skeleton_res.get("edges", []):
        skeleton_edges.append(
            (e["from"], e["to"], e.get("partial_corr"))
        )

# -------------------------
# HUMAN REVIEW POINT (critical)
# -------------------------
filtered_edges, review = apply_human_priors(
    skeleton_edges,
    blacklist=HUMAN_PRIOR_BLACKLIST, # calling method to generate this in config
    whitelist=HUMAN_PRIOR_WHITELIST, # defined in config
)

print("\n=== HUMAN REVIEW SUMMARY ===")
print("Edges removed:", len(review.get("removed", [])))
print("Edges added:", len(review.get("added", [])))

# COMMAND ----------

# -------------------------
# Visualization
# -------------------------

print("Visualizing candidate causal skeleton")
G = visualize_skeleton(filtered_edges)


# COMMAND ----------

# -------------------------
# Step 5: Granger validation
# -------------------------
print("Running Granger tests on candidate edges")
granger_res = granger_check(lagged, filtered_edges)
print("Granger check output", granger_res)


# Export Casual Artifacts
artifacts = {
    "preprocess_meta": preprocess_meta,
    "lag_meta": lag_meta,
    "skeleton": skeleton_res,
    "filtered_edges": filtered_edges,
    "review": review,
    "granger": {f"{a}->{b}": v for (a, b), v in granger_res.items()},
    # "notears": notears_res,
}

lagged.to_csv(f"{path}/causal_metrics_matrix.csv")
with open(f"{path}/causal_artifacts.json", "w") as f:
    json.dump(artifacts, f, default=str)

# Generate ranked candidate list from granger_res
rows = []
for (a,b), info in granger_res.items():
    if info.get("skipped"):
        continue
    p_a_to_b = min(info.get("pvals_a_to_b", [1.0])) if info.get("pvals_a_to_b") else 1.0
    p_b_to_a = min(info.get("pvals_b_to_a", [1.0])) if info.get("pvals_b_to_a") else 1.0
    rows.append({
        "from": a,
        "to": b,
        "p_a_to_b": float(p_a_to_b),
        "a_causes_b": bool(info.get("a_causes_b") is True),
        "p_b_to_a": float(p_b_to_a),
        "b_causes_a": bool(info.get("b_causes_a") is True),
        "min_p": float(min(p_a_to_b, p_b_to_a)),
    })
cand_df = pd.DataFrame(rows).sort_values("min_p")
cand_df.to_csv(f"{path}/causal_candidates_granger.csv", index=False)

# COMMAND ----------

# Save artifacts (uncomment / adjust paths as desired)
artifacts = {
    "preprocess_meta": preprocess_meta,
    "lag_meta": lag_meta,
    "skeleton": skeleton_res,
    "filtered_edges": filtered_edges,
    "review": review,
    "granger": {f"{a}->{b}": v for (a, b), v in granger_res.items()},
    # "notears": notears_res,
}

lagged.to_csv(f"{path}/causal_metrics_matrix.csv")
with open(f"{path}/causal_artifacts.json", "w") as f:
    json.dump(artifacts, f, default=str)

# Generate ranked candidate list from granger_res
rows = []
for (a,b), info in granger_res.items():
    if info.get("skipped"):
        continue
    p_a_to_b = min(info.get("pvals_a_to_b", [1.0])) if info.get("pvals_a_to_b") else 1.0
    p_b_to_a = min(info.get("pvals_b_to_a", [1.0])) if info.get("pvals_b_to_a") else 1.0
    rows.append({
        "from": a,
        "to": b,
        "p_a_to_b": float(p_a_to_b),
        "a_causes_b": bool(info.get("a_causes_b") is True),
        "p_b_to_a": float(p_b_to_a),
        "b_causes_a": bool(info.get("b_causes_a") is True),
        "min_p": float(min(p_a_to_b, p_b_to_a)),
    })
cand_df = pd.DataFrame(rows).sort_values("min_p")
cand_df.to_csv(f"{path}/causal_candidates_granger.csv", index=False)

# COMMAND ----------

# -------------------------
# Step 7: Compute Baseline Statistics for Inference
# -------------------------
print("Computing baseline statistics for inference (step 7)")

def compute_baseline_stats(df):
    """
    Compute baseline statistics for each metric for use in anomaly detection.
    
    Args:
        df: DataFrame where each column is a metric
        
    Returns:
        Dict mapping metric_name -> stats dict
    """
    baseline = {}
    
    for col in df.columns:
        values = df[col].dropna()
        
        if len(values) == 0:
            baseline[col] = {
                'n': 0,
                'mean': None,
                'std': None,
                'median': None,
                'q1': None,
                'q3': None,
                'IQR': None,
                'min': None,
                'max': None
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

baseline_stats = compute_baseline_stats(lagged)
print(f"Computed baseline statistics for {len(baseline_stats)} metrics")

# Save baseline stats
dbutils.fs.put(f"{path}/baseline_stats.json", json.dumps(baseline_stats, indent=2), overwrite=True)
print(f"✓ Saved baseline stats to {path}/baseline_stats.json")

# COMMAND ----------

print("Building adjacency maps for graph traversal (step 8)")

def build_adjacency_maps(edges):
    """
    Build upstream and downstream adjacency lists from edge list.
    
    Args:
        edges: List of (from_node, to_node) tuples or lists
        
    Returns:
        upstream_map: Dict[node -> list of parent nodes]
        downstream_map: Dict[node -> list of child nodes]
    """    
    upstream_map = defaultdict(list)
    downstream_map = defaultdict(list)
    
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            parent, child = edge[0], edge[1]
            
            # Upstream: child -> parents
            upstream_map[child].append(parent)
            
            # Downstream: parent -> children
            downstream_map[parent].append(child)
    
    # Convert to regular dicts for JSON serialization
    return dict(upstream_map), dict(downstream_map)

upstream_map, downstream_map = build_adjacency_maps(filtered_edges)

print(f"Upstream map built: {len(upstream_map)} nodes have parents")
print(f"Downstream map built: {len(downstream_map)} nodes have children")

# Save adjacency maps
dbutils.fs.put(f"{path}/upstream_map.json", json.dumps(upstream_map, indent=2), overwrite=True)
print(f"✓ Saved upstream map to {path}/upstream_map.json")

dbutils.fs.put(f"{path}/downstream_map.json", json.dumps(downstream_map, indent=2), overwrite=True)
print(f"✓ Saved downstream map to {path}/downstream_map.json")

print("\n" + "="*80)
print("✓ CAUSAL DISCOVERY COMPLETE — All artifacts saved for inference")
print("="*80)
print(f"\nSaved artifacts:")
print(f"  - {path}/causal_artifacts.json")
print(f"  - {path}/causal_metrics_matrix.csv")
print(f"  - {path}/causal_candidates_granger.csv")
print(f"  - {path}/baseline_stats.json")
print(f"  - {path}/upstream_map.json")
print(f"  - {path}/downstream_map.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix

# COMMAND ----------

# # NOTERA DEFINATION
# def run_notears(df: pd.DataFrame, l1=0.1, max_iter=100):
#     """
#     Deterministic NOTEARS wrapper using causal_discovery.algos.notears.NoTears only.

#     Assumes `NoTears` from `causal_discovery` is available. If not available
#     returns `notears-unavailable`.
#     """
#     if not has_notears or NoTears is None:
#         return {"method": "notears-unavailable", "adj": None}

#     X = df.values.astype(float)
#     import numpy as _np

#     try:
#         # Prefer no-arg constructor; if it requires positional args, pass
#         # common defaults (rho, alpha, l1_reg).
#         try:
#             model = NoTears()
#         except TypeError:
#             model = NoTears(1.0, 0.0, float(l1))

#         # Fit model (many implementations use fit(X))
#         try:
#             res = model.fit(X)
#         except TypeError:
#             # some versions use fit() without args
#             try:
#                 res = model.fit()
#             except Exception:
#                 res = None

#         # If fit returned adjacency
#         if isinstance(res, (list, tuple, _np.ndarray)):
#             return {"method": "notears", "adj": _np.asarray(res), "columns": df.columns.tolist()}

#         # Otherwise read common attributes
#         for attr in ("W_est", "W", "adj", "W_hat", "weight_matrix"):
#             if hasattr(model, attr):
#                 W = getattr(model, attr)
#                 return {"method": "notears", "adj": _np.asarray(W), "columns": df.columns.tolist()}

#         for fn in ("get_adj", "adj_matrix", "get_W"):
#             if hasattr(model, fn):
#                 try:
#                     W = getattr(model, fn)()
#                     return {"method": "notears", "adj": _np.asarray(W), "columns": df.columns.tolist()}
#                 except Exception:
#                     pass

#         return {"method": "notears-error", "error": "No adjacency found or fit failed", "adj": None}
#     except Exception as e:
#         return {"method": "notears-error", "error": str(e), "adj": None}


# COMMAND ----------

# # # -------------------------
# # # Step 6: NOTEARS Driver
# # # -------------------------
# print("Running NOTEARS")
# notears_res = run_notears(lagged)
# print("NoTears check output", notears_res)

# COMMAND ----------


# ----------------------------
# Human priors (domain constraints)
# ----------------------------
# Notes:
# - We encode two kinds of priors:
#   * Hard blacklist: edges that are not possible by design (e.g., Silver -> Raw)
#   * Hard whitelist: edges we strongly believe must exist (e.g., Raw -> Bronze for survival_rate impacts)

# Programmatic helper: generate blacklist rules based on metric name prefixes
# This helper *does not* run automatically; call it after you load the pivoted matrix
# Example usage inside the orchestration after `metrics_pdf` exists:
#    auto_blacklist = generate_stage_blacklist(metrics_pdf.columns)
#    HUMAN_PRIOR_BLACKLIST.extend(auto_blacklist)

# HUMAN INTERVENTION POINT: 
# - Edit HUMAN_PRIOR_BLACKLIST and HUMAN_PRIOR_WHITELIST above to encode hard constraints before learning.
# - Optionally call generate_stage_blacklist(metrics_pdf.columns) after loading the pivot to auto-populate conservative rules.
# - After the skeleton is learned, review/prune edges at the review/prior step in the orchestration cell.


# [('from_metric', 'to_metric'), ...]
# HUMAN_PRIOR_WHITELIST = [
#     ("raw_input_record_count", "bronze_input_rows"),             # raw record count is the direct input to cleaning -> bronze rows 
#     ("raw_ingestion_duration_sec", "bronze_ingestion_duration_sec"),  # ingestion time propagates downstream
#     ("raw_min_trip_start_ts", "bronze_input_rows"),              # run temporal window limits which raw rows are included 
#     ("raw_max_trip_end_ts", "bronze_output_rows"),               # run temporal end affects which trips survive cleaning  
#     ("raw_distance_mean", "bronze_distance_km_mean"),            # same underlying distance aggregate (unit/derivation change)
#     ("raw_avg_speed_mean", "silver_avg_speed_imputed"),         # avg-speed upstream used for downstream imputation/derived speed metrics  
#     ("raw_unique_units", "bronze_output_rows"),                  # unique unit count influences dedup and output volume  
#     ("raw_null_count_unit_id", "bronze_null_primary_key_rows"),  # null unit_id upstream -> primary-key nulls in bronze 
#     ("bronze_input_rows", "silver_input_data_count"),            # bronze input volume becomes silver's input count 
#     ("bronze_output_rows", "silver_count_after_feature_engineering"),# bronze output flows into feature-engineering counts 
#     ("bronze_survival_rate", "silver_survival_rate"),            # survival fraction defined upstream drives downstream survival stat 
#     ("bronze_distance_km_mean", "silver_avg_speed_imputed"),     # distance/duration aggregates inform speed imputation downstream  
#     ("bronze_duration_mean", "silver_avg_speed_imputed"),        # duration upstream used for avg-speed imputation downstream  
#     ("bronze_ingestion_duration_sec", "silver_ingestion_duration_sec"), # ingestion runtime carries forward 

#     ("bronze_distance_km_mean", "mean_fuel_per_100km"),          # fuel-per-100km KPIs depend on distance aggregates  
#     ("bronze_distance_km_mean", "p50_fuel_per_100km"),           # percentile KPIs depend on distance/fuel aggregates  
#     ("bronze_distance_km_mean", "p95_fuel_per_100km"),           #  

#     ("silver_ml_imputed_fuel_mean", "silver_ml_prediction_mean"),# imputed-fuel mean and model prediction mean are closely related (confidence: medium-high)
#     ("silver_ml_residual_mean", "silver_ml_abs_residual_mean"),  # residual mean and absolute-residual mean are derived from same residuals 
# ]

# metric_cols = spark.table("bms_ds_prod.bms_ds_dasc.temp_raw_metrics").select("metric_name").distinct().rdd.flatMap(lambda x: x).collect()
# # print(metric_cols, type(metric_cols))
# HUMAN_PRIOR_BLACKLIST = generate_stage_blacklist(metric_cols)
# print(HUMAN_PRIOR_BLACKLIST)