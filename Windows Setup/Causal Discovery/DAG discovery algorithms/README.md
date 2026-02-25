# Causal Discovery Pipelines - Local Development Setup

Three causal discovery pipelines (PC-Based, Graphical Lasso, NOTEARS) converted from Databricks to local pandas-based Jupyter notebooks.

## Overview

Three causal discovery approaches implemented as standalone Jupyter notebooks:

1. **Pipeline A: PC-Based** (`Pipeline_A_PC_Based.ipynb`)
   - Uses PC algorithm with configurable alpha
   - Discovers directed relationships via conditional independence tests
   - Output: Directed edges

2. **Pipeline B: Graphical Lasso-Based** (`Pipeline_B_Graphical_Lasso.ipynb`)
   - Uses GraphicalLassoCV for sparse precision matrix estimation
   - Discovers undirected relationships via partial correlations
   - Output: Undirected edges with weights

3. **Pipeline C: NOTEARS-Based** (`Pipeline_C_NOTEARS.ipynb`)
   - Uses NOTEARS algorithm with acyclicity constraints
   - Discovers directed acyclic graph (DAG)
   - Output: Directed acyclic graph with weights

All pipelines:
- Load metrics from `metrics.csv`
- Transform to features matrix format
- Preprocess and select features
- Apply human priors (blacklist/whitelist)
- Visualize results in notebooks
- Save all artifacts to timestamped folders

## Setup Instructions

### 1. Create Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or with conda:
```bash
conda create -n causal-local python=3.9
conda activate causal-local
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas`, `numpy`, `scipy`, `scikit-learn` - Data processing & ML
- `networkx` - Graph visualization
- `causal-learn` - PC algorithm
- `jupyter`, `ipython` - Notebook execution
- `matplotlib`, `seaborn` - Visualization

### 3. Verify Data File

The pipelines expect `metrics.csv` at:
```
Causal Discovery/Metrics data/metrics.csv
```

File should have columns: `date`, `pipeline_stage`, `metric_name`, `metric_value`, `created_at`

### 4. Run Pipelines

**Option A: Sequential execution in Jupyter**
```bash
jupyter notebook
```
Then open and run each pipeline:
1. `Pipeline_A_PC_Based.ipynb`
2. `Pipeline_B_Graphical_Lasso.ipynb`
3. `Pipeline_C_NOTEARS.ipynb`

**Option B: JupyterLab**
```bash
jupyter lab
```

**Option C: Command-line execution**
```bash
jupyter nbconvert --to notebook --execute Pipeline_A_PC_Based.ipynb
```

## Pipeline Architecture

### Data Flow

```
metrics.csv
    ↓
[metrics_to_matrix] → Transform to samples × features matrix
    ↓
[preprocess_metrics_matrix] → Handle missing values, zscore normalize
    ↓
[sophisticated_feature_selection] → Remove low-variance, correlated features
    ↓
[Algorithm] → PC / GraphicalLasso / NOTEARS
    ↓
[apply_blacklist] → Remove forbidden edges by stage rules
    ↓
Artifacts → JSON, CSV, visualizations
```

### Output Structure

Each pipeline creates a timestamped artifact folder:
```
artifacts/
├── PC_Based_20260225_144500/
│   ├── causal_artifacts.json          # Complete pipeline metadata
│   ├── causal_metrics_matrix.csv      # Final feature matrix
│   ├── pc_raw_edges.csv              # Raw edges before filtering
│   ├── pc_causal_edges.csv           # Final edges after blacklist
│   ├── baseline_stats.json           # Mean, std, median per metric
│   ├── upstream_map.json             # Parent → [children] mapping
│   ├── downstream_map.json           # Child → [parents] mapping
│   ├── raw_upstream_map.json
│   └── raw_downstream_map.json
├── Graphical_Lasso_Based_20260225_144600/
│   └── [similar structure]
└── NOTEARS_Based_20260225_144700/
    └── [similar structure]
```

## Configuration

Each pipeline has a Configuration cell with adjustable parameters:

### Common Parameters
```python
MAX_RUNS_TO_PIVOT = 65              # Number of recent dates to use
TARGET_FEATURES = 40-45             # Final feature count
CORRELATION_THRESHOLD = 0.95-0.98   # Remove correlated features
```

### Pipeline A (PC)
```python
PC_ALPHA = 0.05                     # Significance level for CI tests
PC_INDEP_TEST = 'fisherz'          # Independence test method
```

### Pipeline B (Graphical Lasso)
```python
USE_CV = True                       # Use cross-validation for alpha selection
CV_ALPHAS = np.logspace(-3, 1, 20) # Alpha grid for CV
PCOR_THRESHOLD = 0.1               # Min partial correlation to keep edge
```

### Pipeline C (NOTEARS)
```python
LAMBDA1 = 0.0                      # L1 regularization strength
LAMBDA2 = 0.0                      # L2 regularization strength
MAX_ITER = 100                     # Optimization iterations
W_PERCENTILE = 90                  # Keep top 90% weighted edges
```

## Utilities Module

Shared functions in `causal_discovery_utils.py`:

### Data Loading
- `load_metrics_csv()` - Load CSV metrics
- `metrics_to_matrix()` - Transform to features matrix

### Preprocessing
- `preprocess_metrics_matrix()` - Handle missing values, normalize
- `sophisticated_feature_selection()` - Remove low-quality features

### Graph Utilities
- `build_adjacency_maps()` - Create parent/child mappings
- `visualize_skeleton()` - Undirected graph visualization
- `visualize_dag()` - Directed acyclic graph visualization

### Statistics & Artifacts
- `compute_baseline_stats()` - Mean, std, median per metric
- `create_timestamped_artifact_dir()` - Create versioned output folder
- `save_json_artifact()`, `save_csv_artifact()` - Save results

## Key Conversions (Databricks → Pandas)

| Original | Local |
|----------|-------|
| `spark.table()` | `pd.read_csv()` |
| `spark.sql()` | pandas operations |
| `dbutils.fs.mkdirs()` | `os.makedirs()` |
| `dbutils.fs.put()` | file write operations |
| PC algorithm via causal-learn | causal-learn library |
| `sklearn.covariance.GraphicalLassoCV` | Same library |
| NOTEARS custom implementation | Custom class NOTEARSLinear |

## Understanding Results

### PC Algorithm Output
- **Edge Type**: Directed
- **Meaning**: Conditional independence relationships
- **Weight**: Not estimated (set to 1.0)
- **Interpretation**: An edge A→B means A and B are conditionally dependent given observed data

### Graphical Lasso Output
- **Edge Type**: Undirected
- **Meaning**: Partial correlation (conditional dependence)
- **Weight**: Partial correlation coefficient
- **Interpretation**: Strength of relationship after controlling for other variables

### NOTEARS Output
- **Edge Type**: Directed (acyclic)
- **Meaning**: Linear causal relationships
- **Weight**: Regression coefficient
- **Interpretation**: A→B means increasing A by 1 unit increases B by the coefficient value

## Troubleshooting

### "ModuleNotFoundError: No module named 'causal_learn'"
```bash
pip install causal-learn
```

### "FileNotFoundError: metrics.csv"
Verify the path:
```python
METRICS_CSV_PATH = "/path/to/metrics.csv"
```

### No edges discovered
Possible causes:
- Features not sufficiently correlated
- Alpha/threshold too strict
- Features removed during preprocessing
Try:
- Increase `PC_ALPHA` (less strict)
- Decrease `PCOR_THRESHOLD` (more permissive)
- Reduce feature selection aggression

### Memory errors with large datasets
Reduce data size:
```python
MAX_RUNS_TO_PIVOT = 30  # Use fewer runs
TARGET_FEATURES = 20    # Use fewer features
```

### Slow optimization (NOTEARS)
Reduce iterations:
```python
MAX_ITER = 50  # Instead of 100
```

## Development Notes

- All notebooks are fully self-contained
- No Databricks or Spark dependencies
- Each pipeline can run independently
- Metrics CSV schema must match: `[date, pipeline_stage, metric_name, metric_value, created_at]`
- Timestamped output folders prevent overwrites
- Visualizations display in-notebook and save to artifacts

## Performance Notes

- **PC Algorithm**: Complexity O(d³) where d=features, best for d<50
- **Graphical Lasso**: Efficient, scales well to 100+ features
- **NOTEARS**: Slower (optimization-based), good for d<40

For 65 runs × 30-45 features, expect:
- PC: 1-3 minutes
- Graphical Lasso: 30-60 seconds
- NOTEARS: 5-15 minutes

## Next Steps

1. **Update metrics source**: Change `METRICS_CSV_PATH` to production data
2. **Tune parameters**: Adjust alpha/thresholds for your data
3. **Validate results**: Compare graphs across all three pipelines
4. **Integrate with downstream**: Use edge lists for RCA or impact analysis
5. **Schedule runs**: Automate via cron or workflow orchestrator

## Support

For issues:
1. Check notebook cell outputs for error messages
2. Review markdown documentation in each cell
3. Verify metrics CSV structure
4. Check configuration parameters
5. Try reducing data size or features

---

**Last Updated:** February 2026
**Python Version:** 3.9+
**Framework:** Pandas + scikit-learn (no Databricks required)
