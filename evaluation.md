# RCA Evaluation Framework

This document explains how the Unified RCA Evaluation system works, including anomaly detection, candidate scoring, and evaluation metrics.

---

## 1. Multiple Test Dates = Multiple Test Cases

Each date in `TEST_DATES` is treated as an **independent test case**. The system runs the complete pipeline (anomaly detection → ranking → evaluation) separately for each date.

```
TEST_DATES = ["2026-01-16", "2026-01-17"]
         ↓
┌─────────────────────────────────────────────────────────────┐
│ FOR each test_date in TEST_DATES:                           │
│   ├── Load metrics for that date (1 row = 37 metric values) │
│   ├── Look up ground truth for that date                    │
│   │                                                         │
│   └── FOR each graph in GRAPHS_TO_EVALUATE:                 │
│         ├── Detect anomalies                                │
│         ├── Score & rank candidates                         │
│         └── Evaluate against ground truth                   │
│                                                             │
│   → Store results with test_date label                      │
└─────────────────────────────────────────────────────────────┘
```

**Example**: 2 dates × 6 graphs = **12 separate evaluations**, each treating that day's metrics as an independent test case.

---

## 2. Anomaly Detection (Dual-Test Method)

**Simple explanation**: Compare today's metric value against the baseline (historical average from training period).

### Algorithm

```
For each metric:
┌────────────────────────────────────────────────────────────┐
│ TEST 1: Z-Score (assumes normal distribution)              │
│   z = (today's_value - baseline_mean) / baseline_std       │
│   Flag if |z| > 3 (i.e., more than 3 std devs away)        │
├────────────────────────────────────────────────────────────┤
│ TEST 2: IQR (works even for skewed data)                   │
│   lower = Q1 - 1.5 × IQR                                   │
│   upper = Q3 + 1.5 × IQR                                   │
│   Flag if value < lower OR value > upper                   │
├────────────────────────────────────────────────────────────┤
│ RESULT: Anomaly if EITHER test triggers                    │
└────────────────────────────────────────────────────────────┘
```

### Why Dual Tests?

| Test | Strength | When It Works Best |
|------|----------|-------------------|
| **Z-Score** | Catches extreme values | Normally distributed metrics |
| **IQR** | Distribution-free, robust to outliers | Skewed distributions |
| **Combined** | More sensitive, better coverage | All metric types |

### Example

- `raw_null_count_unit_id` baseline: mean=100, std=20
- Today's value: 5000 (40% nulls injected)
- Z-score: (5000 - 100) / 20 = **245** → Way above 3 → **ANOMALY!**

---

## 3. Candidate Scoring & Ranking

Two traversal methods are used depending on the graph type:

### Method A: Downstream Traversal (for PC, NOTEARS - directed graphs)

**Intuition**: *"A true root cause will have many anomalies DOWNSTREAM of it"*

```
For each candidate metric:
┌─────────────────────────────────────────────────────────────┐
│ 1. Start at candidate (e.g., raw_null_count_unit_id)        │
│ 2. Follow edges DOWNSTREAM (parent → child)                 │
│ 3. Count how many anomalies you can reach                   │
│ 4. Closer anomalies = higher score (decay = 0.6 per hop)    │
│ 5. Multiply by edge weight (stronger edges = more impact)   │
│ 6. Bonus ×2 if candidate itself is anomalous                │
└─────────────────────────────────────────────────────────────┘
```

**Scoring Example**:

```
  raw_null_count_unit_id (start)
       ↓ (hop 1, score = 1.0 × 0.6 = 0.6)
  bronze_survival_rate [ANOMALY!] → +0.6
       ↓ (hop 2, score = 0.6 × 0.6 = 0.36)
  silver_join_miss_rate [ANOMALY!] → +0.36
       
  Total score = 0.6 + 0.36 = 0.96
  × 2.0 (self anomaly bonus) = 1.92
```

### Method B: Upstream Traversal (for GraphicalLasso - undirected graph)

**Intuition**: *"Start from each anomaly, trace back, and count which metrics are visited by MANY anomalies"*

```
┌─────────────────────────────────────────────────────────────┐
│ 1. For EACH detected anomaly:                               │
│    - BFS traverse UPSTREAM (toward potential causes)        │
│    - Each visited node gets +1 score (with decay)           │
│                                                             │
│ 2. Nodes visited by MANY anomalies get higher total score   │
│                                                             │
│ 3. True root causes naturally bubble up because they're     │
│    connected to many downstream anomalies                   │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm Selection by Graph Type

| Graph Type | Algorithm | Traversal Method | Edge Weights |
|------------|-----------|------------------|--------------|
| PC | Constraint-based | Upstream | OLS regression coefficients (|β|) |
| NOTEARS | Optimization-based DAG | Upstream | Structural equation coefficients (|W_ij|) |
| GraphicalLasso | Undirected | Upstream | Absolute partial correlations |

### Why Upstream Traversal is Better for RCA

For **Root Cause Analysis**, upstream traversal is more appropriate than downstream because:

| Reason | Explanation |
|--------|-------------|
| **Aligned with RCA intuition** | Debugging naturally starts from symptoms and traces back to cause |
| **Symptom-driven** | We observe anomalies first, then find causes (not the other way around) |
| **Finds common causes** | Metrics upstream of MANY anomalies naturally score higher |
| **More efficient** | Only explores paths connected to actual problems |
| **Matches debugging workflow** | Data engineers trace back from broken metrics |

### Upstream Traversal Example

```
                    [ROOT CAUSE]
                raw_null_count_unit_id
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
bronze_survival_rate    bronze_null_pk_rows
   [ANOMALY]                [ANOMALY]
        ↓                       ↓
silver_join_miss_rate   silver_row_count
   [ANOMALY]                [ANOMALY]
```

**Upstream approach**: 
1. Start from all 4 anomalies
2. Trace back following edges (child → parent)
3. `raw_null_count_unit_id` is visited 4 times (once from each anomaly path)
4. **Highest score → Ranked #1**

---

## 4. How Parent/Child Relationships Are Determined

### Where Does Edge Direction Come From?

The **causal discovery algorithms** automatically determine which metric is the "parent" (cause) and which is the "child" (effect) during the graph learning phase.

### Per-Algorithm Edge Direction

| Algorithm | How Edges Are Oriented | Example |
|-----------|----------------------|---------|
| **NOTEARS** | Weight matrix W where W[i,j] ≠ 0 means **i → j** | If W["raw_nulls", "bronze_survival"] = -0.8, then raw_nulls → bronze_survival |
| **PC** | Statistical tests + Meek orientation rules to determine direction | Conditional independence tests find edges, then rules orient them |
| **GraphicalLasso** | **No direction** - edges are undirected | A—B means A and B are correlated, but we don't know which causes which |

### How Adjacency Maps Are Built

The causal discovery pipelines automatically build `upstream_map` and `downstream_map` from the learned edges:

```python
# For each edge A → B discovered by the algorithm:
upstream_map[B].append(A)    # B's parent is A
downstream_map[A].append(B)  # A's child is B

# Example:
# Edge: raw_null_count_unit_id → bronze_survival_rate
upstream_map["bronze_survival_rate"] = ["raw_null_count_unit_id"]
downstream_map["raw_null_count_unit_id"] = ["bronze_survival_rate"]
```

### What Determines "Top" vs "Bottom" Nodes?

The algorithm learns this from **data patterns** during training:

```
┌─────────────────────────────────────────────────────────────────┐
│ NOTEARS/PC learn edge directions from:                          │
│                                                                 │
│ 1. TEMPORAL PATTERNS                                            │
│    - raw_row_count changes FIRST                                │
│    - bronze_row_count changes AFTER (with lag)                  │
│    → raw causes bronze, not vice versa                          │
│                                                                 │
│ 2. CONDITIONAL INDEPENDENCE                                     │
│    - If A ⊥ C | B (A independent of C given B)                  │
│    - Then B mediates: A → B → C                                 │
│                                                                 │
│ 3. STRUCTURAL CONSTRAINTS                                       │
│    - NOTEARS enforces acyclicity (no loops)                     │
│    - This forces a hierarchical structure                       │
│                                                                 │
│ 4. HUMAN PRIORS (whitelist)                                     │
│    - We explicitly tell it: raw → bronze → silver               │
│    - This guides orientation based on domain knowledge          │
└─────────────────────────────────────────────────────────────────┘
```

### Visual: From Data to Adjacency Maps

```
TRAINING DATA (44 days)              CAUSAL DISCOVERY              ADJACENCY MAPS
┌────────────────────┐              ┌─────────────────┐           ┌─────────────────┐
│ date  | raw | brz  │              │                 │           │ upstream_map:   │
│ Jan 1 | 100 | 95   │   NOTEARS    │  raw ──→ brz    │           │  brz: [raw]     │
│ Jan 2 | 102 | 97   │ ──────────→  │   ↓       ↓     │ ────────→ │  slv: [brz]     │
│ Jan 3 |  98 | 93   │   or PC      │  slv ←── ...    │           │                 │
│ ...   | ...| ...   │              │                 │           │ downstream_map: │
└────────────────────┘              └─────────────────┘           │  raw: [brz]     │
                                                                  │  brz: [slv]     │
                                                                  └─────────────────┘
```

### For Undirected Graphs (GraphicalLasso)

Since GraphicalLasso produces **undirected edges** (A—B), we treat them as bidirectional:

```python
# Edge: A — B (undirected)
upstream_map[A].append(B)    # A's neighbor is B
upstream_map[B].append(A)    # B's neighbor is A
downstream_map[A].append(B)  # Same thing
downstream_map[B].append(A)  # Bidirectional

# Result: Both directions are traversable
# Upstream traversal will explore ALL connected nodes
```

---

## 5. Evaluation Metrics

### Final Ranking

Sort all candidates by score (descending) → **Rank 1 = most likely root cause**

### Metrics Used

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Top-K Accuracy** | 1 if any root in top-K, else 0 | Did we find AT LEAST ONE root cause in top-K? |
| **MRR** | 1 / (rank of first root) | How quickly did we find the first root? |
| **Precision@K** | (# roots in top-K) / K | What % of top-K predictions are correct? |
| **Recall@K** | (# roots in top-K) / total_roots | What % of all true roots did we find in top-K? |

### MRR Interpretation

| MRR Value | Meaning |
|-----------|---------|
| 1.0 | First root at rank 1 (perfect) |
| 0.5 | First root at rank 2 (good) |
| 0.33 | First root at rank 3 (acceptable) |
| 0.1 | First root at rank 10 (poor) |

### Success Criteria

| Metric | Minimum | Target | Failure Indicates |
|--------|---------|--------|-------------------|
| **Top-3 Accuracy** | 0.5 | 1.0 | Root cause not in top-3 → causal graph quality issue |
| **MRR** | 0.33 | 1.0 | First root beyond rank 3 → poor discrimination |
| **Precision@3** | 0.33 | 0.67 | Too many false positives in top-3 |
| **Recall@10** | 1.0 | 1.0 | Missing root causes entirely → detection failure |

---

## 6. Unified vs Old Evaluation Notebook

| Aspect | Old `candidate evaluation.py` | New `unified_rca_evaluation.py` |
|--------|-------------------------------|----------------------------------|
| **Graphs** | Single graph (from old Granger-based pipeline) | 6 graphs (3 algorithms × filtered/raw) |
| **Edge Weights** | `-log10(p_value)` from Granger test | Algorithm-specific: `abs_weight`, `abs_partial_corr` |
| **Test Cases** | Single date hardcoded | Multiple dates in one run |
| **Comparison** | No comparison | Side-by-side comparison across all graphs |
| **Scoring Methods** | Upstream only (hardcoded) | Adaptive (downstream for DAGs, upstream for undirected) |
| **Ground Truth** | Hardcoded in code | Configurable dictionary with date mapping |

**Key Improvement**: The unified notebook runs the **same evaluation logic** but across **multiple graphs simultaneously**, allowing direct comparison of which algorithm works best.

---

## 7. Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIFIED EVALUATION FLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TEST_DATES: [2026-01-16, 2026-01-17]                               │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐                                                │
│  │ For each date:  │                                                │
│  │  Load metrics   │ ← 37 metrics for that day                      │
│  │  Get ground     │ ← {raw_null_count_unit_id, raw_unique_units}   │
│  │   truth         │                                                │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ For each graph (PC/GLasso/NOTEARS × filtered/raw):         │     │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │     │
│  │  │ 1. Detect   │ → │ 2. Score &  │ → │ 3. Evaluate │       │     │
│  │  │   anomalies │   │    rank     │   │   vs truth  │       │     │
│  │  │ (Z+IQR)     │   │ (BFS+decay) │   │ (MRR,Top-K) │       │     │
│  │  └─────────────┘   └─────────────┘   └─────────────┘       │     │
│  └────────────────────────────────────────────────────────────┘     │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ COMPARISON: Which graph has best MRR/Top-3/Recall?          │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Configuration Quick Reference

### Selecting Graphs to Evaluate

```python
GRAPHS_TO_EVALUATE = {
    # PC Algorithm
    "pc_filtered": True,      # PC with human priors
    "pc_raw": False,          # PC without human priors
    
    # GraphicalLasso
    "glasso_filtered": True,  # GraphicalLasso with priors
    "glasso_raw": False,      # GraphicalLasso without priors
    
    # NOTEARS
    "notears_filtered": True, # NOTEARS with priors
    "notears_raw": False,     # NOTEARS without priors
}
```

### Common Usage Patterns

| Use Case | Configuration |
|----------|--------------|
| Test single graph | Set only one to `True` |
| Compare filtered vs raw | Set `pc_filtered: True, pc_raw: True` |
| Compare all algorithms (filtered only) | Set all `*_filtered` to `True` |
| Full comparison (all 6 graphs) | Set all to `True` |

### Tunable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Z_SCORE_THRESHOLD` | 3.0 | Flag if \|z-score\| exceeds this |
| `IQR_MULTIPLIER` | 1.5 | Tukey's rule for outlier bounds |
| `MAX_DEPTH` | 3 | Maximum BFS traversal depth |
| `DECAY_FACTOR` | 0.6 | Score decay per hop (60% retained) |
| `SELF_ANOMALY_BONUS` | 2.0 | Multiplier if candidate is anomalous |
